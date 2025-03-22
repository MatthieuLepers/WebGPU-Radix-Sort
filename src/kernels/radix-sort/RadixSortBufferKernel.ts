import { AbstractRadixSortKernel, DispatchData, type AbstractRadixSortKernelOptions } from './AbstractRadixSortKernel';
import radixSortSource from '../../shaders/RadixSort';
import radixSortSourceLocalShuffle from '../../shaders/optimizations/RadixSortLocalShuffle';
import reorderSource from '../../shaders/RadixSortReorder';
import { findOptimalDispatchSize, removeValues } from '../../utils';
import { KernelPipelineDefinition } from '../AbstractKernel';
import { CheckSortBufferKernel } from '../check-sort/CheckSortBufferKernel';

export interface BufferKernel {
  keys: GPUBuffer;
  values?: GPUBuffer;
}

export interface RadixSortKernelOptions extends AbstractRadixSortKernelOptions<BufferKernel> {
  localShuffle?: boolean;
}

export class RadixSortBufferKernel extends AbstractRadixSortKernel<BufferKernel> {
  declare localShuffle: boolean;

  constructor(options: RadixSortKernelOptions) {
    super(options);
    this.localShuffle = options.localShuffle ?? false;

    this.buffers.keys = options.data.keys;
    if (options.data.values) {
      this.buffers.values = options.data.values;
    }

    this.createShaderModules();
    this.createPipelines();
  }

  get hasValues(): boolean {
    return !!this.data.values;
  }

  protected get blockSumSource(): string {
    const source = this.localShuffle
      ? radixSortSourceLocalShuffle
      : radixSortSource('buffer')
    ;
    return this.hasValues ? source : removeValues(source);
  }

  protected get reorderSource(): string {
    return this.hasValues ? reorderSource('buffer') : removeValues(reorderSource('buffer'));
  }

  protected createResources(): void {
    // Keys and values double buffering
    this.buffers.tmpKeys = this.device.createBuffer({
      label: 'radix-sort-tmp-keys',
      size: this.count * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    if (this.hasValues) {
      this.buffers.tmpValues = this.device.createBuffer({
        label: 'radix-sort-tmp-values',
        size: this.count * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      });
    }

    // Local Prefix Sum buffer (1 element per item)
    this.buffers.localPrefixSum = this.device.createBuffer({
      label: 'radix-sort-local-prefix-sum',
      size: this.count * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
  }

  protected getPassInData(even: boolean): BufferKernel {
    return {
      keys: even ? this.buffers.keys : this.buffers.tmpKeys,
      values: even ? this.buffers.values : this.buffers.tmpValues,
    };
  }

  protected getPassOutData(even: boolean): BufferKernel {
    return {
      keys: even ? this.buffers.tmpKeys : this.buffers.keys,
      values: even ? this.buffers.tmpValues : this.buffers.values,
    };
  }

  protected createBlockSumPipeline(inData: BufferKernel, bit: number): KernelPipelineDefinition {
    const bindGroupLayout = this.device.createBindGroupLayout({
      label: 'radix-sort-block-sum',
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.COMPUTE,
          buffer: {
            type: this.localShuffle
              ? 'storage' as GPUBufferBindingType
              : 'read-only-storage' as GPUBufferBindingType,
          }
        },
        {
          binding: 1,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: 'storage' as GPUBufferBindingType },
        },
        {
          binding: 2,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: 'storage' as GPUBufferBindingType },
        },
        ...(this.localShuffle && this.hasValues ? [{
          binding: 3,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: 'storage' as GPUBufferBindingType },
        }] : []),
      ],
    });

    const bindGroup = this.device.createBindGroup({
      layout: bindGroupLayout,
      entries: [
        {
          binding: 0,
          resource: { buffer: inData.keys },
        },
        {
          binding: 1,
          resource: { buffer: this.buffers.localPrefixSum },
        },
        {
          binding: 2,
          resource: { buffer: this.buffers.prefixBlockSum },
        },
        // "Local shuffle" optimization needs access to the values buffer
        ...(this.localShuffle && this.hasValues ? [{
          binding: 3,
          resource: { buffer: inData.values! },
        }] : []),
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
    this.kernels.checkSortFull = new CheckSortBufferKernel({
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
    this.kernels.checkSortFast = new CheckSortBufferKernel({
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
    this.kernels.checkSortReset = new CheckSortBufferKernel({
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

  protected createReorderPipeline(inData: BufferKernel, outData: BufferKernel, bit: number): KernelPipelineDefinition {
    const bindGroupLayout = this.device.createBindGroupLayout({
      label: 'radix-sort-reorder',
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: 'read-only-storage' as GPUBufferBindingType },
        },
        {
          binding: 1,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: 'storage' as GPUBufferBindingType },
        },
        {
          binding: 2,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: 'read-only-storage' as GPUBufferBindingType },
        },
        {
          binding: 3,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: 'read-only-storage' as GPUBufferBindingType },
        },
        ...(this.hasValues ? [
          {
            binding: 4,
            visibility: GPUShaderStage.COMPUTE,
            buffer: { type: 'read-only-storage' as GPUBufferBindingType },
          },
          {
            binding: 5,
            visibility: GPUShaderStage.COMPUTE,
            buffer: { type: 'storage' as GPUBufferBindingType },
          },
        ] : []),
      ],
    });

    const bindGroup = this.device.createBindGroup({
      layout: bindGroupLayout,
      entries: [
        {
          binding: 0,
          resource: { buffer: inData.keys },
        },
        {
          binding: 1,
          resource: { buffer: outData.keys },
        },
        {
          binding: 2,
          resource: { buffer: this.buffers.localPrefixSum },
        },
        {
          binding: 3,
          resource: { buffer: this.buffers.prefixBlockSum },
        },
        ...(this.hasValues ? [
          {
            binding: 4,
            resource: { buffer: inData.values! },
          },
          {
            binding: 5,
            resource: { buffer: outData.values! },
          },
        ] : []),
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
