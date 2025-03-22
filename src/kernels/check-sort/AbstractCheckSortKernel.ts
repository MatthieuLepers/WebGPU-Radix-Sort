import { AbstractKernel, type AbstractKernelOptions } from '../AbstractKernel';
import { findOptimalDispatchSize, WorkgroupSize } from '../../utils';

export interface AbstractCheckSortKernelOptions<T> extends AbstractKernelOptions {
  data: T;
  result: GPUBuffer;
  original: GPUBuffer;
  isSorted: GPUBuffer;
  start?: number;
  mode?: 'full' | 'fast' | 'reset';
}

export abstract class AbstractCheckSortKernel<T> extends AbstractKernel {
  declare start: number;

  declare mode: 'full' | 'fast' | 'reset';

  public buffers: Record<string, GPUBuffer> = {};

  constructor(options: AbstractCheckSortKernelOptions<T>) {
    super(options);
    this.start = options.start ?? 0;
    this.mode = options.mode ?? 'full';

    this.buffers.result = options.result;
    this.buffers.original = options.original;
    this.buffers.isSorted = options.isSorted;
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

  protected abstract createPassesRecursive(data: T, count: number, passIndex: number): void;

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
