import { findOptimalDispatchSize } from '../utils';
import prefixSumSource from '../shaders/PrefixSum';
import prefixSumSourceNoBankConflict from '../shaders/optimizations/PrefixSumNoBankConflict';
import { AbstractKernel, type AbstractKernelOptions } from './AbstractKernel';

interface IPrefixSumKernelOptions extends AbstractKernelOptions {
  data: GPUBuffer;
  avoidBankConflicts?: boolean;
}

export class PrefixSumKernel extends AbstractKernel {
  /**
   * Perform a parallel prefix sum on the given data buffer
   * 
   * Based on "Parallel Prefix Sum (Scan) with CUDA"
   * https://www.eecs.umich.edu/courses/eecs570/hw/parprefix.pdf
   * 
   * @param {GPUDevice} device
   * @param {number} count - Max number of elements to process
   * @param {WorkgroupSize} workgroupSize - Workgroup size in x and y dimensions. (x * y) must be a power of two
   * @param {GPUBuffer} data - Buffer containing the data to process
   * @param {boolean} avoidBankConflicts - Use the "Avoid bank conflicts" optimization from the original publication
   */
  constructor({
    device,
    count,
    workgroupSize = { x: 16, y: 16 },
    data,
    avoidBankConflicts = false
  }: IPrefixSumKernelOptions) {
    super({ device, count, workgroupSize });

    if (Math.log2(this.threadsPerWorkgroup) % 1 !== 0) {
      throw new Error(`workgroupSize.x * workgroupSize.y must be a power of two. (current: ${this.threadsPerWorkgroup})`);
    }

    this.shaderModules.prefixSum = this.device.createShaderModule({
      label: 'prefix-sum',
      code: avoidBankConflicts ? prefixSumSourceNoBankConflict : prefixSumSource,
    })

    this.createPassRecursive(data, count);
  }

  createPassRecursive(data: GPUBuffer, count: number) {
    // Find best dispatch x and y dimensions to minimize unused threads
    const workgroupCount = Math.ceil(count / this.itemsPerWorkgroup);
    const dispatchSize = findOptimalDispatchSize(this.device, workgroupCount);

    // Create buffer for block sums
    const blockSumBuffer = this.device.createBuffer({
      label: 'prefix-sum-block-sum',
      size: workgroupCount * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });

    // Create bind group and pipeline layout
    const bindGroupLayout = this.device.createBindGroupLayout({
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: 'storage' as GPUBufferBindingType },
        },
        {
          binding: 1,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: 'storage' as GPUBufferBindingType },
        },
      ],
    });

    const bindGroup = this.device.createBindGroup({
      label: 'prefix-sum-bind-group',
      layout: bindGroupLayout,
      entries: [
        {
          binding: 0,
          resource: { buffer: data },
        },
        {
          binding: 1,
          resource: { buffer: blockSumBuffer },
        },
      ],
    });

    const pipelineLayout = this.device.createPipelineLayout({
      bindGroupLayouts: [ bindGroupLayout ],
    });

    // Per-workgroup (block) prefix sum
    const scanPipeline = this.device.createComputePipeline({
      label: 'prefix-sum-scan-pipeline',
      layout: pipelineLayout,
      compute: {
        module: this.shaderModules.prefixSum,
        entryPoint: 'reduce_downsweep',
        constants: {
          'WORKGROUP_SIZE_X': this.workgroupSize.x,
          'WORKGROUP_SIZE_Y': this.workgroupSize.y,
          'THREADS_PER_WORKGROUP': this.threadsPerWorkgroup,
          'ITEMS_PER_WORKGROUP': this.itemsPerWorkgroup,
          'ELEMENT_COUNT': count,
        },
      },
    });

    this.pipelines.push({ pipeline: scanPipeline, bindGroup, dispatchSize });

    if (workgroupCount > 1) {
      // Prefix sum on block sums
      this.createPassRecursive(blockSumBuffer, workgroupCount);

      // Add block sums to local prefix sums
      const blockSumPipeline = this.device.createComputePipeline({
        label: 'prefix-sum-add-block-pipeline',
        layout: pipelineLayout,
        compute: {
          module: this.shaderModules.prefixSum,
          entryPoint: 'add_block_sums',
          constants: {
            'WORKGROUP_SIZE_X': this.workgroupSize.x,
            'WORKGROUP_SIZE_Y': this.workgroupSize.y,
            'THREADS_PER_WORKGROUP': this.threadsPerWorkgroup,
            'ELEMENT_COUNT': count,
          },
        },
      });

      this.pipelines.push({ pipeline: blockSumPipeline, bindGroup, dispatchSize });
    }
  }

  getDispatchChain() {
    return this.pipelines.flatMap((p) => [ p.dispatchSize!.x, p.dispatchSize!.y, 1 ]);
  }

  /**
   * Encode the prefix sum pipeline into the current pass.
   * If dispatchSizeBuffer is provided, the dispatch will be indirect (dispatchWorkgroupsIndirect)
   *
   * @param {GPUComputePassEncoder} pass
   * @param {GPUBuffer} dispatchSizeBuffer - (optional) Indirect dispatch buffer
   * @param {number} offset - (optional) Offset in bytes in the dispatch buffer. Default: 0
   */
  dispatch(pass: GPUComputePassEncoder, dispatchSizeBuffer?: GPUBuffer, offset: number = 0) {
    this.pipelines.forEach(({ pipeline, bindGroup, dispatchSize }, i) => {
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bindGroup);

      if (!dispatchSizeBuffer) {
        pass.dispatchWorkgroups(dispatchSize!.x, dispatchSize!.y, 1);
      } else {
        pass.dispatchWorkgroupsIndirect(dispatchSizeBuffer, offset + i * 3 * 4);
      }
    });
  }
}
