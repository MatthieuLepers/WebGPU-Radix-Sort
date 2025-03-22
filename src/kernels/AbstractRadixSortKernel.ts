import { AbstractKernel, type AbstractKernelOptions } from 'src/AbstractKernel';
import { CheckSortKernel } from 'src/CheckSortKernel';
import { PrefixSumKernel } from 'src/PrefixSumKernel';
import { createBufferFromData, type DispatchSize, findOptimalDispatchSize } from 'src/utils';

export interface DispatchData {
  initialDispatch: Array<number>;
  dispatchSizesFull: Array<number>;
  checkSortFastCount: number;
  checkSortFullCount: number;
  startFull: number;
}

export interface AbstractRadixSortKernelOptions<T> extends AbstractKernelOptions {
  data: T;
  bitCount?: number;
  checkOrder?: boolean;
  avoidBankConflicts?: boolean;
}

export abstract class AbstractRadixSortKernel<T> extends AbstractKernel {
  declare data: T;

  public bitCount: number = 32;

  public checkOrder: boolean = false;

  public avoidBankConflicts: boolean = false;

  public buffers: Record<string, GPUBuffer> = {};

  private kernels: {
    prefixSum?: PrefixSumKernel;
    checkSortReset?: CheckSortKernel;
    checkSortFast?: CheckSortKernel;
    checkSortFull?: CheckSortKernel;
  } = {};

  private dispatchSize: DispatchSize = {
    x: 1,
    y: 1,
  };

  private dispatchOffsets = {
    radixSort: 0,
    checkSortFast: 3 * 4,
    prefixSum: 6 * 4
  };

  private initialDispatch: Array<number> = [];

  constructor(options: AbstractRadixSortKernelOptions<T>) {
    super(options);

    this.#createShaderModules();

    this.#createPipelines();
  }

  get prefixBlockWorkgroupCount(): number {
    return 4 * this.workgroupCount;
  }

  abstract get hasValues(): boolean;

  protected abstract get blockSumSource(): string;

  protected abstract get reorderSource(): string;

  #createShaderModules() {
    this.shaderModules.blockSum = this.device.createShaderModule({
      label: 'radix-sort-block-sum',
      code: this.blockSumSource,
    });
    this.shaderModules.reorder = this.device.createShaderModule({
      label: 'radix-sort-reorder',
      code: this.reorderSource,
    });
  }

  #createPipelines() {
    // Block prefix sum kernel
    this.#createPrefixSumKernel();

    // Indirect dispatch buffers
    const dispatchData = this.#calculateDispatchSizes();

    // GPU resources (buffers / textures)
    this.createResources();
    this.#createCheckSortBuffers(dispatchData);

    this.#createCheckSortKernels(dispatchData);

    // Radix sort passes for every 2 bits
    for (let bit = 0; bit < this.bitCount; bit += 2) {
      // Swap buffers every pass
      const even = (bit % 4 === 0);
      const inData = this.getPassInData(even);
      const outData = this.getPassOutData(even);

      // Compute local prefix sums and block sums
      const blockSumPipeline = this.#createBlockSumPipeline(inData, bit);

      // Reorder keys and values
      const reorderPipeline = this.#createReorderPipeline(inData, outData, bit);

      this.pipelines.push(blockSumPipeline, reorderPipeline);
    }
  }

  #createPrefixSumKernel() {
    // Prefix Block Sum buffer (4 element per workgroup)
    const prefixBlockSumBuffer = this.device.createBuffer({
      label: 'radix-sort-prefix-block-sum',
      size: this.prefixBlockWorkgroupCount * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });

    // Create block prefix sum kernel
    const prefixSumKernel = new PrefixSumKernel({
      device: this.device,
      data: prefixBlockSumBuffer,
      count: this.prefixBlockWorkgroupCount,
      workgroupSize: this.workgroupSize,
      avoidBankConflicts: this.avoidBankConflicts,
    });

    this.kernels.prefixSum = prefixSumKernel;
    this.buffers.prefixBlockSum = prefixBlockSumBuffer;
  }

  #calculateDispatchSizes(): DispatchData {
    // Radix sort dispatch size
    const dispatchSize = findOptimalDispatchSize(this.device, this.workgroupCount);

    // Prefix sum dispatch sizes
    const prefixSumDispatchSize = this.kernels.prefixSum!.getDispatchChain();

    // Check sort element count (fast/full)
    const checkSortFastCount = Math.min(this.count, this.threadsPerWorkgroup * 4);
    const checkSortFullCount = this.count - checkSortFastCount;
    const startFull = checkSortFastCount - 1;

    // Check sort dispatch sizes
    const dispatchSizesFast = CheckSortKernel.findOptimalDispatchChain(this.device, checkSortFastCount, this.workgroupSize);
    const dispatchSizesFull = CheckSortKernel.findOptimalDispatchChain(this.device, checkSortFullCount, this.workgroupSize);

    // Initial dispatch sizes
    const initialDispatch = [
      dispatchSize.x, dispatchSize.y, 1, // Radix Sort + Reorder
      ...dispatchSizesFast.slice(0, 3),  // Check sort fast
      ...prefixSumDispatchSize,          // Prefix Sum
    ];

    this.dispatchSize = dispatchSize;
    this.initialDispatch = initialDispatch;

    return {
      initialDispatch,
      dispatchSizesFull,
      checkSortFastCount,
      checkSortFullCount,
      startFull,
    };
  }

  protected abstract createResources(): void;

  protected abstract getPassInData(even: boolean): T;

  protected abstract getPassOutData(even: boolean): T;

  #createCheckSortBuffers(dispatchData: DispatchData) {
    // Only create indirect dispatch buffers when checkOrder optimization is enabled
    if (!this.checkOrder) {
      return;
    }

    // Dispatch sizes (radix sort, check sort, prefix sum)
    this.buffers.dispatchSize = createBufferFromData({
      device: this.device,
      label: 'radix-sort-dispatch-size',
      data: dispatchData.initialDispatch,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.INDIRECT,
    });
    this.buffers.originalDispatchSize = createBufferFromData({
      device: this.device,
      label: 'radix-sort-dispatch-size-original',
      data: dispatchData.initialDispatch,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });

    // Dispatch sizes (full sort)
    this.buffers.checkSortFullDispatchSize = createBufferFromData({
      label: 'check-sort-full-dispatch-size',
      device: this.device,
      data: dispatchData.dispatchSizesFull,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.INDIRECT,
    });
    this.buffers.originalCheckSortFullDispatchSize = createBufferFromData({
      label: 'check-sort-full-dispatch-size-original',
      device: this.device,
      data: dispatchData.dispatchSizesFull,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });

    // Flag to tell if the data is sorted
    this.buffers.isSorted = createBufferFromData({
      label: 'is-sorted',
      device: this.device,
      data: [0],
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
  }
}
