import checkSortSource from '../../shaders/CheckSort';
import type { TextureKernel } from '../radix-sort/RadixSortTextureKernel';
import { bufferToTexture } from '../../utils';
import { AbstractCheckSortKernel, type AbstractCheckSortKernelOptions } from './AbstractCheckSortKernel';

export interface CheckSortTextureKernelOptions extends AbstractCheckSortKernelOptions<TextureKernel> {
}

export class CheckSortTextureKernel extends AbstractCheckSortKernel<TextureKernel> {
  public textures: Record<string, GPUTexture> = {};

  public outputs: Array<GPUTexture> = [];

  constructor(options: CheckSortTextureKernelOptions) {
    super(options);

    this.textures.read = options.data.texture;

    this.createPassesRecursive(options.data, this.count);
  }

  createPassesRecursive(data: TextureKernel, count: number, passIndex: number = 0) {
    const workgroupCount = Math.ceil(count / this.threadsPerWorkgroup);

    const isFirstPass = !passIndex;
    const isLastPass = workgroupCount <= 1;

    const label = `check-sort-${this.mode}-${passIndex}`;

    const outputBuffer = isLastPass ? this.buffers.result : this.device.createBuffer({
      label,
      size: workgroupCount * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    const outputTexture = bufferToTexture(this.device, outputBuffer, 'rg32uint');

    const bindGroupLayout = this.device.createBindGroupLayout({
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
            access: 'read-write',
            format: 'rg32uint',
            viewDimension: '2d',
          },
        },
        // Last pass bindings
        ...(isLastPass ? [{
          binding: 2,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: 'read-only-storage' as GPUBufferBindingType },
        }, {
          binding: 3,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: 'storage' as GPUBufferBindingType }
        }] : []),
      ],
    });

    const bindGroup = this.device.createBindGroup({
      layout: bindGroupLayout,
      entries: [
        {
          binding: 0,
          resource: data.texture.createView(),
        },
        {
          binding: 1,
          resource: outputTexture.createView(),
        },
        // Last pass buffers
        ...(isLastPass ? [{
          binding: 2,
          resource: { buffer: this.buffers.original },
        }, {
          binding: 3,
          resource: { buffer: this.buffers.isSorted },
        }] : []),
      ],
    });

    const pipelineLayout = this.device.createPipelineLayout({
      bindGroupLayouts: [bindGroupLayout],
    });

    const elementCount = isFirstPass ? this.start + count : count;
    const startElement = isFirstPass ? this.start : 0;

    const checkSortPipeline = this.device.createComputePipeline({
      layout: pipelineLayout,
      compute: {
        module: this.device.createShaderModule({
          label,
          code: checkSortSource(isFirstPass, isLastPass, this.mode, 'texture'),
        }),
        entryPoint: this.mode == 'reset' ? 'reset' : 'check_sort',
        constants: {
          'ELEMENT_COUNT': elementCount,
          'WORKGROUP_SIZE_X': this.workgroupSize.x,
          'WORKGROUP_SIZE_Y': this.workgroupSize.y,
          ...(this.mode !== 'reset' && {
            'THREADS_PER_WORKGROUP': this.threadsPerWorkgroup,
            'START_ELEMENT': startElement,
          }),
        },
      },
    });

    this.outputs.push(outputTexture);
    this.pipelines.push({ pipeline: checkSortPipeline, bindGroup });

    if (!isLastPass) {
      this.createPassesRecursive({ texture: outputTexture }, workgroupCount, passIndex + 1);
    }
  }
}