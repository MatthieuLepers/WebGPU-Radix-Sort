import { findOptimalDispatchSize } from './utils';
import type { WorkgroupSize } from './utils';
import checkSortSource from './shaders/CheckSort';
import { AbstractKernel, type AbstractKernelOptions } from './AbstractKernel';

interface ICheckSortKernelOptions extends AbstractKernelOptions {
  data: GPUBuffer;
  result: GPUBuffer;
  original: GPUBuffer;
  isSorted: GPUBuffer;
  start?: number;
  mode?: 'full' | 'fast' | 'reset';
}

export class CheckSortKernel extends AbstractKernel {
  public start: number;

  public mode: 'full' | 'fast' | 'reset';

  private buffers: Record<string, GPUBuffer> = {};

  public outputs: Array<GPUBuffer> = [];

  /**
   * CheckSortKernel - Performs a parralel reduction to check if an array is sorted.
   * 
   * @param {GPUDevice} device
   * @param {number} count - The number of elements to check
   * @param {WorkgroupSize} workgroupSize - The workgroup size in x and y dimensions
   * @param {GPUBuffer} data - The buffer containing the data to check
   * @param {GPUBuffer} result - The result dispatch size buffer
   * @param {GPUBuffer} original - The original dispatch size buffer
   * @param {GPUBuffer} isSorted - 1-element buffer to store whether the array is sorted
   * @param {number} start - The index to start checking from
   * @param {boolean} mode - The type of check sort kernel ('reset', 'fast', 'full')
   */
  constructor({
    device,
    count,
    workgroupSize = { x: 16, y: 16 },
    data,
    result,
    original,
    isSorted,
    start = 0,
    mode = 'full',
  }: ICheckSortKernelOptions) {
    super({ device, count, workgroupSize });
    this.start = start;
    this.mode = mode;

    this.buffers = {
      data,
      result,
      original,
      isSorted,
    };

    this.createPassesRecursive(data, count);
  }

  // Find the best dispatch size for each pass to minimize unused workgroups
  static findOptimalDispatchChain(device: GPUDevice, itemCount: number, workgroupSize: WorkgroupSize) {
    const threadsPerWorkgroup = workgroupSize.x * workgroupSize.y;
    const sizes = [];

    do {
      // Number of workgroups required to process all items
      const targetWorkgroupCount = Math.ceil(itemCount / threadsPerWorkgroup);

      // Optimal dispatch size and updated workgroup count
      const dispatchSize = findOptimalDispatchSize(device, targetWorkgroupCount);

      sizes.push(dispatchSize.x, dispatchSize.y, 1);
      itemCount = targetWorkgroupCount;
    } while (itemCount > 1);

    return sizes;
  }

  createPassesRecursive(buffer: GPUBuffer, count: number, passIndex: number = 0) {
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
          resource: { buffer },
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
          code: checkSortSource(isFirstPass, isLastPass, this.mode),
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
      this.createPassesRecursive(outputBuffer, workgroupCount, passIndex + 1);
    }
  }

  dispatch(pass: GPUComputePassEncoder, dispatchSizeBuffer?: GPUBuffer, offset: number = 0) {
    this.pipelines.forEach(({ pipeline, bindGroup }, i) => {
      const dispatchIndirect = this.mode !== 'reset' && (this.mode === 'full' || i < this.pipelines.length - 1);

      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bindGroup);

      if (dispatchIndirect && dispatchSizeBuffer) {
        pass.dispatchWorkgroupsIndirect(dispatchSizeBuffer, offset + i * 3 * 4);
      } else {
        // Only the reset kernel and the last dispatch of the fast check kernel are constant to (1, 1, 1)
        pass.dispatchWorkgroups(1, 1, 1);
      }
    });
  }
}
