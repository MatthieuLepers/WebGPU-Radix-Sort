import { PrefixSumKernel } from './PrefixSumKernel';
import { CheckSortKernel } from './CheckSortKernel';
import { createBufferFromData, findOptimalDispatchSize } from './utils';
import type { WorkgroupSize, DispatchSize } from './utils';
import radixSortSource from './shaders/RadixSort';
import radixSortSourceLocalShuffle from './shaders/optimizations/RadixSortLocalShuffle';
import reorderSource from './shaders/RadixSortReorder';
import { AbstractKernel } from './AbstractKernel';

interface IRadixSortKernelOptions {
  device: GPUDevice;
  count: number;
  workgroupSize: WorkgroupSize;
  texture?: GPUTexture;
  keys?: GPUBuffer;
  values?: GPUBuffer;
  bitCount: number;
  checkOrder?: boolean;
  localShuffle?: boolean;
  avoidBankConflicts?: boolean;
}

interface IDispatchData {
  initialDispatch: Array<number>;
  dispatchSizesFull: Array<number>;
  checkSortFastCount: number;
  checkSortFullCount: number;
  startFull: number;
}

export class RadixSortKernel extends AbstractKernel {
  public bitCount: number;

  public checkOrder: boolean = false;

  public localShuffle: boolean = false;

  public avoidBankConflicts: boolean = false;

  private prefixBlockWorkgroupCount: number;

  private hasValues: boolean;

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

  private kernels: {
    prefixSum?: PrefixSumKernel;
    checkSort?: {
      reset: CheckSortKernel;
      fast: CheckSortKernel;
      full: CheckSortKernel;
    };
  } = {};

  public buffers: Record<string, GPUBuffer | undefined> = {};

  /**
   * Perform a parallel radix sort on the GPU given a buffer of keys and (optionnaly) values
   * Note: The buffers are sorted in-place.
   * 
   * Based on "Fast 4-way parallel radix sorting on GPUs"
   * https://www.sci.utah.edu/~csilva/papers/cgf.pdf]
   * 
   * @param {GPUDevice} device
   * @param {number} count - Number of elements to sort
   * @param {WorkgroupSize} workgroupSize - Workgroup size in x and y dimensions. (x * y) must be a power of two
   * @param {GPUBuffer} keys - Buffer containing the keys to sort
   * @param {GPUBuffer} values - (optional) Buffer containing the associated values
   * @param {number} bitCount - Number of bits per element (default: 32)
   * @param {boolean} checkOrder - Enable "order checking" optimization. Can improve performance if the data needs to be sorted in real-time and doesn't change much. (default: false)
   * @param {boolean} localShuffle - Enable "local shuffling" optimization for the radix sort kernel (default: false)
   * @param {boolean} avoidBankConflicts - Enable "avoiding bank conflicts" optimization for the prefix sum kernel (default: false)
   */
  constructor({
    device,
    count,
    workgroupSize = { x: 16, y: 16 },
    texture,
    keys,
    values,
    bitCount = 32,
    checkOrder = false,
    localShuffle = false,
    avoidBankConflicts = false,
  }: IRadixSortKernelOptions) {
    super({ device, count, workgroupSize });
    if (!device) throw new Error('No device provided');
    if (!keys && !texture) throw new Error('No keys buffer or texture provided');
    if (!Number.isInteger(count) || count <= 0) throw new Error('Invalid count parameter');
    if (!Number.isInteger(bitCount) || bitCount <= 0 || bitCount > 32) throw new Error(`Invalid bitCount parameter: ${bitCount}`);
    if (!Number.isInteger(workgroupSize.x) || !Number.isInteger(workgroupSize.y)) throw new Error('Invalid workgroupSize parameter');
    if (bitCount % 4 != 0) throw new Error('bitCount must be a multiple of 4');

    this.bitCount = bitCount;
    this.checkOrder = checkOrder;
    this.localShuffle = localShuffle;
    this.avoidBankConflicts = avoidBankConflicts;

    this.prefixBlockWorkgroupCount = 4 * this.workgroupCount;

    this.hasValues = !!values || !!texture; // Is the values buffer or input texture provided ?

    this.buffers = { keys, values }; // GPUBuffers

    // Create shader modules from wgsl code
    this.createShaderModules();

    // Create multi-pass pipelines
    this.createPipelines();
  }

  private createShaderModules() {
    // Remove every occurence of "values" in the shader code if values buffer is not provided
    const removeValues = (source: string) => source
      .split('\n')
      .filter((line) => !line.toLowerCase().includes('values'))
      .join('\n')
    ;

    const blockSumSource = this.localShuffle
      ? radixSortSourceLocalShuffle
      : radixSortSource(this.buffers.keys!)
    ;

    this.shaderModules = {
      blockSum: this.device.createShaderModule({
        label: 'radix-sort-block-sum',
        code: this.hasValues ? blockSumSource : removeValues(blockSumSource),
      }),
      reorder: this.device.createShaderModule({
        label: 'radix-sort-reorder',
        code: this.hasValues ? reorderSource : removeValues(reorderSource),
      }),
    };
  }

  private createPipelines() {
    // Block prefix sum kernel
    this.createPrefixSumKernel();

    // Indirect dispatch buffers
    const dispatchData = this.calculateDispatchSizes();

    // GPU buffers
    this.createBuffers();
    this.createCheckSortBuffers(dispatchData);

    // Check sort kernels
    this.createCheckSortKernels(dispatchData);

    // Radix sort passes for every 2 bits
    for (let bit = 0; bit < this.bitCount; bit += 2) {
      // Swap buffers every pass
      const even = (bit % 4 == 0);
      const inKeys = even ? this.buffers.keys : this.buffers.tmpKeys;
      const inValues = even ? this.buffers.values : this.buffers.tmpValues;
      const outKeys = even ? this.buffers.tmpKeys : this.buffers.keys;
      const outValues = even ? this.buffers.tmpValues : this.buffers.values;

      // Compute local prefix sums and block sums
      const blockSumPipeline = this.createBlockSumPipeline(inKeys!, inValues!, bit);

      // Reorder keys and values
      const reorderPipeline = this.createReorderPipeline(inKeys!, inValues!, outKeys!, outValues!, bit);

      this.pipelines.push(blockSumPipeline, reorderPipeline);
    }
  }

  private createPrefixSumKernel() {
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

  private calculateDispatchSizes(): IDispatchData {
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

    // Dispatch offsets in main buffer
    this.dispatchOffsets = {
      radixSort: 0,
      checkSortFast: 3 * 4,
      prefixSum: 6 * 4,
    };

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

  private createBuffers() {
    // Keys and values double buffering
    const tmpKeysBuffer = this.device.createBuffer({
      label: 'radix-sort-tmp-keys',
      size: this.count * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    const tmpValuesBuffer = !this.hasValues ? undefined : this.device.createBuffer({
      label: 'radix-sort-tmp-values',
      size: this.count * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });

    // Local Prefix Sum buffer (1 element per item)
    const localPrefixSumBuffer = this.device.createBuffer({
      label: 'radix-sort-local-prefix-sum',
      size: this.count * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });

    this.buffers.tmpKeys = tmpKeysBuffer;
    this.buffers.tmpValues = tmpValuesBuffer;
    this.buffers.localPrefixSum = localPrefixSumBuffer;
  }

  private createCheckSortBuffers(dispatchData: IDispatchData) {
    // Only create indirect dispatch buffers when checkOrder optimization is enabled
    if (!this.checkOrder) {
      return;
    }

    // Dispatch sizes (radix sort, check sort, prefix sum)
    const dispatchBuffer = createBufferFromData({
      device: this.device,
      label: 'radix-sort-dispatch-size',
      data: dispatchData.initialDispatch,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.INDIRECT,
    });
    const originalDispatchBuffer = createBufferFromData({
      device: this.device,
      label: 'radix-sort-dispatch-size-original',
      data: dispatchData.initialDispatch,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });

    // Dispatch sizes (full sort)
    const checkSortFullDispatchBuffer = createBufferFromData({
      label: 'check-sort-full-dispatch-size',
      device: this.device,
      data: dispatchData.dispatchSizesFull,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.INDIRECT,
    });
    const checkSortFullOriginalDispatchBuffer = createBufferFromData({
      label: 'check-sort-full-dispatch-size-original',
      device: this.device,
      data: dispatchData.dispatchSizesFull,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });

    // Flag to tell if the data is sorted
    const isSortedBuffer = createBufferFromData({
      label: 'is-sorted',
      device: this.device,
      data: [0],
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });

    this.buffers.dispatchSize = dispatchBuffer;
    this.buffers.originalDispatchSize = originalDispatchBuffer;
    this.buffers.checkSortFullDispatchSize = checkSortFullDispatchBuffer;
    this.buffers.originalCheckSortFullDispatchSize = checkSortFullOriginalDispatchBuffer;
    this.buffers.isSorted = isSortedBuffer;
  }

  private createCheckSortKernels(checkSortPartitionData: IDispatchData) {
    if (!this.checkOrder) {
      return;
    }

    const { checkSortFastCount, checkSortFullCount, startFull } = checkSortPartitionData

    // Create the full pass
    const checkSortFull = new CheckSortKernel({
      mode: 'full',
      device: this.device,
      data: this.buffers.keys!,
      result: this.buffers.dispatchSize!,
      original: this.buffers.originalDispatchSize!,
      isSorted: this.buffers.isSorted!,
      count: checkSortFullCount,
      start: startFull,
      workgroupSize: this.workgroupSize,
    });

    // Create the fast pass
    const checkSortFast = new CheckSortKernel({
      mode: 'fast',
      device: this.device,
      data: this.buffers.keys!,
      result: this.buffers.checkSortFullDispatchSize!,
      original: this.buffers.originalCheckSortFullDispatchSize!,
      isSorted: this.buffers.isSorted!,
      count: checkSortFastCount,
      workgroupSize: this.workgroupSize,
    });

    const initialDispatchElementCount = this.initialDispatch.length / 3;

    if (checkSortFast.threadsPerWorkgroup < checkSortFull.pipelines.length || checkSortFull.threadsPerWorkgroup < initialDispatchElementCount) {
      console.warn(`Warning: workgroup size is too small to enable check sort optimization, disabling...`);
      this.checkOrder = false;
      return;
    }

    // Create the reset pass
    const checkSortReset = new CheckSortKernel({
      mode: 'reset',
      device: this.device,
      data: this.buffers.keys!,
      original: this.buffers.originalDispatchSize!,
      result: this.buffers.dispatchSize!,
      isSorted: this.buffers.isSorted!,
      count: initialDispatchElementCount,
      workgroupSize: findOptimalDispatchSize(this.device, initialDispatchElementCount),
    });

    this.kernels.checkSort = {
      reset: checkSortReset,
      fast: checkSortFast,
      full: checkSortFull,
    };
  }

  private createBlockSumPipeline(inKeys: GPUBuffer, inValues: GPUBuffer, bit: number) {
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
          resource: { buffer: inKeys },
        },
        {
          binding: 1,
          resource: { buffer: this.buffers.localPrefixSum! },
        },
        {
          binding: 2,
          resource: { buffer: this.buffers.prefixBlockSum! },
        },
        // "Local shuffle" optimization needs access to the values buffer
        ...(this.localShuffle && this.hasValues ? [{
          binding: 3,
          resource: { buffer: inValues }
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

  createReorderPipeline(inKeys: GPUBuffer, inValues: GPUBuffer, outKeys: GPUBuffer, outValues: GPUBuffer, bit: number) {
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
          resource: { buffer: inKeys },
        },
        {
          binding: 1,
          resource: { buffer: outKeys },
        },
        {
          binding: 2,
          resource: { buffer: this.buffers.localPrefixSum! },
        },
        {
          binding: 3,
          resource: { buffer: this.buffers.prefixBlockSum! },
        },
        ...(this.hasValues ? [
          {
            binding: 4,
            resource: { buffer: inValues },
          },
          {
            binding: 5,
            resource: { buffer: outValues },
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

  /**
   * Encode all pipelines into the current pass
   * 
   * @param {GPUComputePassEncoder} pass 
   */
  dispatch(pass: GPUComputePassEncoder) {
    if (!this.checkOrder) {
      this.#dispatchPipelines(pass);
    } else {
      this.#dispatchPipelinesIndirect(pass);
    }
  }

  /**
   * Dispatch workgroups from CPU args
   */
  #dispatchPipelines(pass: GPUComputePassEncoder) {
    for (let i = 0; i < this.bitCount / 2; i += 1) {
      const blockSumPipeline = this.pipelines[i * 2];
      const reorderPipeline = this.pipelines[i * 2 + 1];

      // Compute local prefix sums and block sums
      pass.setPipeline(blockSumPipeline.pipeline);
      pass.setBindGroup(0, blockSumPipeline.bindGroup);
      pass.dispatchWorkgroups(this.dispatchSize.x, this.dispatchSize.y, 1);

      // Compute block sums prefix sum
      this.kernels.prefixSum!.dispatch(pass);

      // Reorder keys and values
      pass.setPipeline(reorderPipeline.pipeline);
      pass.setBindGroup(0, reorderPipeline.bindGroup);
      pass.dispatchWorkgroups(this.dispatchSize.x, this.dispatchSize.y, 1);
    }
  }

  /**
   * Dispatch workgroups from indirect GPU buffers (used when checkOrder is enabled)
   */
  #dispatchPipelinesIndirect(pass: GPUComputePassEncoder) {
    // Reset the `dispatch` and `is_sorted` buffers
    this.kernels.checkSort!.reset.dispatch(pass);

    for (let i = 0; i < this.bitCount / 2; i++) {
      const blockSumPipeline = this.pipelines[i * 2];
      const reorderPipeline = this.pipelines[i * 2 + 1];

      if (i % 2 == 0) {
        // Check if the data is sorted every 2 passes
        this.kernels.checkSort!.fast.dispatch(pass, this.buffers.dispatchSize, this.dispatchOffsets.checkSortFast);
        this.kernels.checkSort!.full.dispatch(pass, this.buffers.checkSortFullDispatchSize);
      }

      // Compute local prefix sums and block sums
      pass.setPipeline(blockSumPipeline.pipeline);
      pass.setBindGroup(0, blockSumPipeline.bindGroup);
      pass.dispatchWorkgroupsIndirect(this.buffers.dispatchSize!, this.dispatchOffsets.radixSort);

      // Compute block sums prefix sum
      this.kernels.prefixSum!.dispatch(pass, this.buffers.dispatchSize, this.dispatchOffsets.prefixSum);

      // Reorder keys and values
      pass.setPipeline(reorderPipeline.pipeline);
      pass.setBindGroup(0, reorderPipeline.bindGroup);
      pass.dispatchWorkgroupsIndirect(this.buffers.dispatchSize!, this.dispatchOffsets.radixSort);
    }
  }
}
