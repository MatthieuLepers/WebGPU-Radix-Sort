import checkSortSource from '../../shaders/CheckSort';
import { BufferKernel } from '../radix-sort/RadixSortBufferKernel';
import { AbstractCheckSortKernel, AbstractCheckSortKernelOptions } from './AbstractCheckSortKernel';

export interface CheckSortBufferKernelOptions extends AbstractCheckSortKernelOptions<BufferKernel> {
}

export class CheckSortBufferKernel extends AbstractCheckSortKernel<BufferKernel> {
  public outputs: Array<GPUBuffer> = [];

  constructor(options: CheckSortBufferKernelOptions) {
    super(options);

    this.buffers.data = options.data.keys;

    this.createPassesRecursive(options.data, this.count);
  }

  createPassesRecursive(data: BufferKernel, count: number, passIndex: number = 0) {
    const workgroupCount = Math.ceil(count / this.threadsPerWorkgroup);

    const isFirstPass = !passIndex;
    const isLastPass = workgroupCount <= 1;

    const label = `check-sort-${this.mode}-${passIndex}`;

    const outputBuffer = isLastPass ? this.buffers.result : this.device.createBuffer({
      label,
      size: workgroupCount * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });

    const bindGroupLayout = this.device.createBindGroupLayout({
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
          resource: { buffer: data.keys },
        },
        {
          binding: 1,
          resource: { buffer: outputBuffer },
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
          code: checkSortSource(isFirstPass, isLastPass, this.mode, 'buffer'),
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

    this.outputs.push(outputBuffer);
    this.pipelines.push({ pipeline: checkSortPipeline, bindGroup });

    if (!isLastPass) {
      this.createPassesRecursive({ keys: outputBuffer }, workgroupCount, passIndex + 1);
    }
  }
}