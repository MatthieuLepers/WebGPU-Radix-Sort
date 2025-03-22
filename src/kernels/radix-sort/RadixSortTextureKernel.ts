import { AbstractRadixSortKernel, type DispatchData, type AbstractRadixSortKernelOptions } from './AbstractRadixSortKernel';
import radixSortSource from '../../shaders/RadixSort';
import reorderSource from '../../shaders/RadixSortReorder';
import type { KernelPipelineDefinition } from '../AbstractKernel';
import { CheckSortTextureKernel } from '../check-sort/CheckSortTextureKernel';
import { findOptimalDispatchSize } from '../../utils';

export interface TextureKernel {
  texture: GPUTexture;
}

export interface RadixSortKernelOptions extends AbstractRadixSortKernelOptions<TextureKernel> {
}

export class RadixSortTextureKernel extends AbstractRadixSortKernel<TextureKernel> {
  public textures: Record<string, GPUTexture> = {};

  constructor(options: RadixSortKernelOptions) {
    super(options);

    this.textures.read = options.data.texture;

    this.createShaderModules();
    this.createPipelines();
  }

  get hasValues(): boolean {
    return true;
  }

  protected get blockSumSource(): string {
    return radixSortSource('texture');
  }

  protected get reorderSource(): string {
    return reorderSource('texture');
  }

  protected createResources(): void {
    // Write texture
    this.textures.write = this.device.createTexture({
      size: {
        width: this.textures.read.width,
        height: this.textures.read.height,
      },
      format: this.textures.read.format,
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.COPY_SRC,
    });

    // Local Prefix Sum texture
    this.textures.localPrefixSum = this.device.createTexture({
      size: {
        width: this.textures.read.width,
        height: this.textures.read.height,
      },
      format: 'r32uint',
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.COPY_SRC,
    });
  }

  protected getPassInData(even: boolean): TextureKernel {
    return {
      texture: even ? this.textures.read : this.textures.write,
    };
  }

  protected getPassOutData(even: boolean): TextureKernel {
    return {
      texture: even ? this.textures.write : this.textures.read,
    };
  }

  protected createBlockSumPipeline(inData: TextureKernel, bit: number) {
    const bindGroupLayout = this.device.createBindGroupLayout({
      label: 'radix-sort-block-sum',
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.COMPUTE,
          storageTexture: {
            access: 'read-only',
            format: 'rg32uint',
            viewDimension: '2d',
          },
        },
        {
          binding: 1,
          visibility: GPUShaderStage.COMPUTE,
          storageTexture: {
            access: 'write-only',
            format: 'r32uint',
            viewDimension: '2d',
          },
        },
        {
          binding: 2,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: 'storage' as GPUBufferBindingType },
        },
      ],
    });

    const bindGroup = this.device.createBindGroup({
      layout: bindGroupLayout,
      entries: [
        {
          binding: 0,
          resource: inData.texture.createView(),
        },
        {
          binding: 1,
          resource: this.textures.localPrefixSum.createView(),
        },
        {
          binding: 2,
          resource: { buffer: this.buffers.prefixBlockSum },
        },
      ],
    });

    const pipelineLayout = this.device.createPipelineLayout({
      bindGroupLayouts: [bindGroupLayout],
    });

    const blockSumPipeline = this.device.createComputePipeline({
      label: 'radix-sort-block-sum',
      layout: pipelineLayout,
      compute: {
        module: this.shaderModules.blockSum,
        entryPoint: 'radix_sort',
        constants: {
          'WORKGROUP_SIZE_X': this.workgroupSize.x,
          'WORKGROUP_SIZE_Y': this.workgroupSize.y,
          'WORKGROUP_COUNT': this.workgroupCount,
          'THREADS_PER_WORKGROUP': this.threadsPerWorkgroup,
          'ELEMENT_COUNT': this.count,
          'CURRENT_BIT': bit,
        },
      },
    });

    return {
      pipeline: blockSumPipeline,
      bindGroup,
    };
  }

  protected createCheckSortKernels(dispatchData: DispatchData) {
    if (!this.checkOrder) return;

    const { checkSortFastCount, checkSortFullCount, startFull } = dispatchData;

    // Create the full pass
    this.kernels.checkSortFull = new CheckSortTextureKernel({
      mode: 'full',
      device: this.device,
      data: this.data,
      result: this.buffers.dispatchSize,
      original: this.buffers.originalDispatchSize,
      isSorted: this.buffers.isSorted!,
      count: checkSortFullCount,
      start: startFull,
      workgroupSize: this.workgroupSize,
    });

    // Create the fast pass
    this.kernels.checkSortFast = new CheckSortTextureKernel({
      mode: 'fast',
      device: this.device,
      data: this.data,
      result: this.buffers.checkSortFullDispatchSize,
      original: this.buffers.originalCheckSortFullDispatchSize,
      isSorted: this.buffers.isSorted,
      count: checkSortFastCount,
      workgroupSize: this.workgroupSize,
    });

    const initialDispatchElementCount = this.initialDispatch.length / 3;

    if (this.kernels.checkSortFast!.threadsPerWorkgroup < this.kernels.checkSortFull!.pipelines.length || this.kernels.checkSortFull!.threadsPerWorkgroup < initialDispatchElementCount) {
      console.warn(`Warning: workgroup size is too small to enable check sort optimization, disabling...`);
      this.checkOrder = false;
      return;
    }

    // Create the reset pass
    this.kernels.checkSortReset = new CheckSortTextureKernel({
      mode: 'reset',
      device: this.device,
      data: this.data,
      original: this.buffers.originalDispatchSize,
      result: this.buffers.dispatchSize,
      isSorted: this.buffers.isSorted,
      count: initialDispatchElementCount,
      workgroupSize: findOptimalDispatchSize(this.device, initialDispatchElementCount),
    });
  }

  protected createReorderPipeline(inData: TextureKernel, outData: TextureKernel, bit: number): KernelPipelineDefinition {
    const bindGroupLayout = this.device.createBindGroupLayout({
      label: 'radix-sort-reorder',
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.COMPUTE,
          storageTexture: {
            access: 'read-only',
            format: 'rg32uint',
            viewDimension: '2d',
          },
        },
        {
          binding: 1,
          visibility: GPUShaderStage.COMPUTE,
          storageTexture: {
            access: 'write-only',
            format: 'rg32uint',
            viewDimension: '2d',
          },
        },
        {
          binding: 2,
          visibility: GPUShaderStage.COMPUTE,
          storageTexture: {
            access: 'read-write',
            format: 'r32uint',
            viewDimension: '2d',
          },
        },
        {
          binding: 3,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: 'read-only-storage' as GPUBufferBindingType },
        },
      ],
    });

    const bindGroup = this.device.createBindGroup({
      layout: bindGroupLayout,
      entries: [
        {
          binding: 0,
          resource: inData.texture.createView(),
        },
        {
          binding: 1,
          resource: outData.texture.createView(),
        },
        {
          binding: 2,
          resource: this.textures.localPrefixSum.createView(),
        },
        {
          binding: 3,
          resource: { buffer: this.buffers.prefixBlockSum },
        },
      ],
    });

    const pipelineLayout = this.device.createPipelineLayout({
      bindGroupLayouts: [bindGroupLayout],
    });

    const reorderPipeline = this.device.createComputePipeline({
      label: 'radix-sort-reorder',
      layout: pipelineLayout,
      compute: {
        module: this.shaderModules.reorder,
        entryPoint: 'radix_sort_reorder',
        constants: {
          'WORKGROUP_SIZE_X': this.workgroupSize.x,
          'WORKGROUP_SIZE_Y': this.workgroupSize.y,
          'WORKGROUP_COUNT': this.workgroupCount,
          'THREADS_PER_WORKGROUP': this.threadsPerWorkgroup,
          'ELEMENT_COUNT': this.count,
          'CURRENT_BIT': bit,
        },
      },
    });

    return {
      pipeline: reorderPipeline,
      bindGroup,
    };
  }
}
