import type { DispatchSize, WorkgroupSize } from '../utils';

export interface AbstractKernelOptions {
  device: GPUDevice;
  count: number;
  workgroupSize?: WorkgroupSize;
}

export interface KernelPipelineDefinition {
  pipeline: GPUComputePipeline;
  bindGroup: GPUBindGroup;
  dispatchSize?: DispatchSize;
}

export abstract class AbstractKernel {
  public options: AbstractKernelOptions;

  declare device: GPUDevice;

  declare count: number;

  public workgroupSize: WorkgroupSize = {
    x: 16,
    y: 16,
  };

  public pipelines: Array<KernelPipelineDefinition> = [];

  protected shaderModules: Record<string, GPUShaderModule> = {};

  constructor(options: AbstractKernelOptions) {
    this.options = options;
    Object.keys(options).forEach((key) => {
      Object.defineProperty(this, key, {
        get: () => this.options[key as keyof AbstractKernelOptions],
        set: (val) => { this.options[key as keyof AbstractKernelOptions] = val; },
      });
    });
  }

  get workgroupCount(): number {
    return Math.ceil(this.count / this.threadsPerWorkgroup);
  }

  get threadsPerWorkgroup(): number {
    return this.workgroupSize.x * this.workgroupSize.y
  }

  get itemsPerWorkgroup(): number {
    return 2 * this.threadsPerWorkgroup;
  }

  abstract dispatch(
    passEncoder: GPUComputePassEncoder,
    dispatchSizeBuffer?: GPUBuffer,
    offset?: number,
  ): void;
}
