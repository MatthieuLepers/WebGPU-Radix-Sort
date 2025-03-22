(function(l,c){typeof exports=="object"&&typeof module<"u"?c(exports):typeof define=="function"&&define.amd?define(["exports"],c):(l=typeof globalThis<"u"?globalThis:l||self,c(l.RadixSort={}))})(this,function(l){"use strict";class c{options;workgroupSize={x:16,y:16};pipelines=[];shaderModules={};constructor(e){this.options=e,Object.keys(e).forEach(t=>{Object.defineProperty(this,t,{get:()=>this.options[t],set:i=>{this.options[t]=i}})})}get workgroupCount(){return Math.ceil(this.count/this.threadsPerWorkgroup)}get threadsPerWorkgroup(){return this.workgroupSize.x*this.workgroupSize.y}get itemsPerWorkgroup(){return 2*this.threadsPerWorkgroup}}function p(o,e){const t={x:e,y:1};if(e>o.limits.maxComputeWorkgroupsPerDimension){const i=Math.floor(Math.sqrt(e)),r=Math.ceil(e/i);t.x=i,t.y=r}return t}function _({device:o,label:e,data:t,usage:i=0}){const r=o.createBuffer({label:e,usage:i,size:t.length*4,mappedAtCreation:!0});return new Uint32Array(r.getMappedRange()).set(t),r.unmap(),r}function D(o,e){const t=Math.min(8192,e.size),i=Math.ceil(e.size/t),r=o.createTexture({size:{width:t,height:i},format:"r32uint",usage:GPUTextureUsage.STORAGE_BINDING|GPUTextureUsage.COPY_DST|GPUTextureUsage.COPY_SRC}),s=o.createCommandEncoder();return s.copyBufferToTexture({buffer:e},{texture:r},[r.width,r.height,r.depthOrArrayLayers]),o.queue.submit([s.finish()]),r}const E=o=>o.split(`
`).filter(e=>!e.toLowerCase().includes("values")).join(`
`),k=`
@group(0) @binding(0) var<storage, read_write> items: array<u32>;
@group(0) @binding(1) var<storage, read_write> blockSums: array<u32>;

override WORKGROUP_SIZE_X: u32;
override WORKGROUP_SIZE_Y: u32;
override THREADS_PER_WORKGROUP: u32;
override ITEMS_PER_WORKGROUP: u32;
override ELEMENT_COUNT: u32;

var<workgroup> temp: array<u32, 2u * ITEMS_PER_WORKGROUP>;

@compute @workgroup_size(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, 1)
fn reduce_downsweep(
  @builtin(workgroup_id) w_id: vec3<u32>,
  @builtin(num_workgroups) w_dim: vec3<u32>,
  @builtin(local_invocation_index) TID: u32, // Local thread ID
) {
  let WORKGROUP_ID = w_id.x + w_id.y * w_dim.x;
  let WID = WORKGROUP_ID * THREADS_PER_WORKGROUP;
  let GID = WID + TID; // Global thread ID

  let ELM_TID = TID * 2u; // Element pair local ID
  let ELM_GID = GID * 2u; // Element pair global ID

  // Load input to shared memory
  temp[ELM_TID]      = select(items[ELM_GID], 0u, ELM_GID >= ELEMENT_COUNT);
  temp[ELM_TID + 1u] = select(items[ELM_GID + 1u], 0u, ELM_GID + 1u >= ELEMENT_COUNT);

  var offset: u32 = 1u;

  // Up-sweep (reduce) phase
  for (var d: u32 = ITEMS_PER_WORKGROUP >> 1u; d > 0u; d >>= 1u) {
    workgroupBarrier();

    if (TID < d) {
      var ai: u32 = offset * (ELM_TID + 1u) - 1u;
      var bi: u32 = offset * (ELM_TID + 2u) - 1u;
      temp[bi] += temp[ai];
    }

    offset *= 2u;
  }

    // Save workgroup sum and clear last element
  if (TID == 0u) {
    let last_offset = ITEMS_PER_WORKGROUP - 1u;

    blockSums[WORKGROUP_ID] = temp[last_offset];
    temp[last_offset] = 0u;
  }

  // Down-sweep phase
  for (var d: u32 = 1u; d < ITEMS_PER_WORKGROUP; d *= 2u) {
    offset >>= 1u;
    workgroupBarrier();

    if (TID < d) {
      var ai: u32 = offset * (ELM_TID + 1u) - 1u;
      var bi: u32 = offset * (ELM_TID + 2u) - 1u;

      let t: u32 = temp[ai];
      temp[ai] = temp[bi];
      temp[bi] += t;
    }
  }
  workgroupBarrier();

  // Copy result from shared memory to global memory
  if (ELM_GID >= ELEMENT_COUNT) {
    return;
  }
  items[ELM_GID] = temp[ELM_TID];

  if (ELM_GID + 1u >= ELEMENT_COUNT) {
    return;
  }
  items[ELM_GID + 1u] = temp[ELM_TID + 1u];
}

@compute @workgroup_size(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, 1)
fn add_block_sums(
  @builtin(workgroup_id) w_id: vec3<u32>,
  @builtin(num_workgroups) w_dim: vec3<u32>,
  @builtin(local_invocation_index) TID: u32, // Local thread ID
) {
  let WORKGROUP_ID = w_id.x + w_id.y * w_dim.x;
  let WID = WORKGROUP_ID * THREADS_PER_WORKGROUP;
  let GID = WID + TID; // Global thread ID

  let ELM_ID = GID * 2u;

  if (ELM_ID >= ELEMENT_COUNT) {
    return;
  }

  let blockSum = blockSums[WORKGROUP_ID];

  items[ELM_ID] += blockSum;

  if (ELM_ID + 1u >= ELEMENT_COUNT) {
    return;
  }

  items[ELM_ID + 1u] += blockSum;
}
`,v=`
@group(0) @binding(0) var<storage, read_write> items: array<u32>;
@group(0) @binding(1) var<storage, read_write> blockSums: array<u32>;

override WORKGROUP_SIZE_X: u32;
override WORKGROUP_SIZE_Y: u32;
override THREADS_PER_WORKGROUP: u32;
override ITEMS_PER_WORKGROUP: u32;
override ELEMENT_COUNT: u32;

const NUM_BANKS: u32 = 32u;
const LOG_NUM_BANKS: u32 = 5u;

fn get_offset(offset: u32) -> u32 {
  // return offset >> LOG_NUM_BANKS; // Conflict-free
  return (offset >> NUM_BANKS) + (offset >> (2u * LOG_NUM_BANKS)); // Zero bank conflict
}

var<workgroup> temp: array<u32, 2u * ITEMS_PER_WORKGROUP>;

@compute @workgroup_size(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, 1)
fn reduce_downsweep(
  @builtin(workgroup_id) w_id: vec3<u32>,
  @builtin(num_workgroups) w_dim: vec3<u32>,
  @builtin(local_invocation_index) TID: u32, // Local thread ID
) {
  let WORKGROUP_ID = w_id.x + w_id.y * w_dim.x;
  let WID = WORKGROUP_ID * THREADS_PER_WORKGROUP;
  let GID = WID + TID; // Global thread ID

  let ELM_TID = TID * 2u; // Element pair local ID
  let ELM_GID = GID * 2u; // Element pair global ID

  // Load input to shared memory
  let ai: u32 = TID;
  let bi: u32 = TID + (ITEMS_PER_WORKGROUP >> 1u);
  let s_ai = ai + get_offset(ai);
  let s_bi = bi + get_offset(bi);
  let g_ai = ai + WID * 2u;
  let g_bi = bi + WID * 2u;
  temp[s_ai] = select(items[g_ai], 0u, g_ai >= ELEMENT_COUNT);
  temp[s_bi] = select(items[g_bi], 0u, g_bi >= ELEMENT_COUNT);

  var offset: u32 = 1u;

  // Up-sweep (reduce) phase
  for (var d: u32 = ITEMS_PER_WORKGROUP >> 1u; d > 0u; d >>= 1u) {
    workgroupBarrier();

    if (TID < d) {
      var ai: u32 = offset * (ELM_TID + 1u) - 1u;
      var bi: u32 = offset * (ELM_TID + 2u) - 1u;
      ai += get_offset(ai);
      bi += get_offset(bi);
      temp[bi] += temp[ai];
    }

    offset *= 2u;
  }

  // Save workgroup sum and clear last element
  if (TID == 0u) {
    var last_offset = ITEMS_PER_WORKGROUP - 1u;
    last_offset += get_offset(last_offset);

    blockSums[WORKGROUP_ID] = temp[last_offset];
    temp[last_offset] = 0u;
  }

  // Down-sweep phase
  for (var d: u32 = 1u; d < ITEMS_PER_WORKGROUP; d *= 2u) {
    offset >>= 1u;
    workgroupBarrier();

    if (TID < d) {
      var ai: u32 = offset * (ELM_TID + 1u) - 1u;
      var bi: u32 = offset * (ELM_TID + 2u) - 1u;
      ai += get_offset(ai);
      bi += get_offset(bi);

      let t: u32 = temp[ai];
      temp[ai] = temp[bi];
      temp[bi] += t;
    }
  }
  workgroupBarrier();

  // Copy result from shared memory to global memory
  if (g_ai < ELEMENT_COUNT) {
    items[g_ai] = temp[s_ai];
  }
  if (g_bi < ELEMENT_COUNT) {
    items[g_bi] = temp[s_bi];
  }
}

@compute @workgroup_size(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, 1)
fn add_block_sums(
  @builtin(workgroup_id) w_id: vec3<u32>,
  @builtin(num_workgroups) w_dim: vec3<u32>,
  @builtin(local_invocation_index) TID: u32, // Local thread ID
) {
  let WORKGROUP_ID = w_id.x + w_id.y * w_dim.x;
  let WID = WORKGROUP_ID * THREADS_PER_WORKGROUP;
  let GID = WID + TID; // Global thread ID

  let ELM_ID = GID * 2u;

  if (ELM_ID >= ELEMENT_COUNT) {
    return;
  }

  let blockSum = blockSums[WORKGROUP_ID];

  items[ELM_ID] += blockSum;

  if (ELM_ID + 1u >= ELEMENT_COUNT) {
    return;
  }

  items[ELM_ID + 1u] += blockSum;
}
`;class x extends c{constructor({device:e,count:t,workgroupSize:i={x:16,y:16},data:r,avoidBankConflicts:s=!1}){if(super({device:e,count:t,workgroupSize:i}),Math.log2(this.threadsPerWorkgroup)%1!==0)throw new Error(`workgroupSize.x * workgroupSize.y must be a power of two. (current: ${this.threadsPerWorkgroup})`);this.shaderModules.prefixSum=this.device.createShaderModule({label:"prefix-sum",code:s?v:k}),this.createPassRecursive(r,t)}createPassRecursive(e,t){const i=Math.ceil(t/this.itemsPerWorkgroup),r=p(this.device,i),s=this.device.createBuffer({label:"prefix-sum-block-sum",size:i*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST}),u=this.device.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},{binding:1,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}}]}),a=this.device.createBindGroup({label:"prefix-sum-bind-group",layout:u,entries:[{binding:0,resource:{buffer:e}},{binding:1,resource:{buffer:s}}]}),n=this.device.createPipelineLayout({bindGroupLayouts:[u]}),d=this.device.createComputePipeline({label:"prefix-sum-scan-pipeline",layout:n,compute:{module:this.shaderModules.prefixSum,entryPoint:"reduce_downsweep",constants:{WORKGROUP_SIZE_X:this.workgroupSize.x,WORKGROUP_SIZE_Y:this.workgroupSize.y,THREADS_PER_WORKGROUP:this.threadsPerWorkgroup,ITEMS_PER_WORKGROUP:this.itemsPerWorkgroup,ELEMENT_COUNT:t}}});if(this.pipelines.push({pipeline:d,bindGroup:a,dispatchSize:r}),i>1){this.createPassRecursive(s,i);const f=this.device.createComputePipeline({label:"prefix-sum-add-block-pipeline",layout:n,compute:{module:this.shaderModules.prefixSum,entryPoint:"add_block_sums",constants:{WORKGROUP_SIZE_X:this.workgroupSize.x,WORKGROUP_SIZE_Y:this.workgroupSize.y,THREADS_PER_WORKGROUP:this.threadsPerWorkgroup,ELEMENT_COUNT:t}}});this.pipelines.push({pipeline:f,bindGroup:a,dispatchSize:r})}}getDispatchChain(){return this.pipelines.flatMap(e=>[e.dispatchSize.x,e.dispatchSize.y,1])}dispatch(e,t,i=0){this.pipelines.forEach(({pipeline:r,bindGroup:s,dispatchSize:u},a)=>{e.setPipeline(r),e.setBindGroup(0,s),t?e.dispatchWorkgroupsIndirect(t,i+a*3*4):e.dispatchWorkgroups(u.x,u.y,1)})}}class h extends c{buffers={};constructor(e){super(e),this.start=e.start??0,this.mode=e.mode??"full",this.buffers.result=e.result,this.buffers.original=e.original,this.buffers.isSorted=e.isSorted}static findOptimalDispatchChain(e,t,i){const r=i.x*i.y,s=[];do{const u=Math.ceil(t/r),a=p(e,u);s.push(a.x,a.y,1),t=u}while(t>1);return s}dispatch(e,t,i=0){this.pipelines.forEach(({pipeline:r,bindGroup:s},u)=>{const a=this.mode!=="reset"&&(this.mode==="full"||u<this.pipelines.length-1);e.setPipeline(r),e.setBindGroup(0,s),a&&t?e.dispatchWorkgroupsIndirect(t,i+u*3*4):e.dispatchWorkgroups(1,1,1)})}}class U extends c{buffers={};kernels={};dispatchSize={x:1,y:1};dispatchOffsets={radixSort:0,checkSortFast:3*4,prefixSum:6*4};initialDispatch=[];constructor(e){super(e),this.bitCount=e.bitCount??32,this.checkOrder=e.checkOrder??!1,this.avoidBankConflicts=e.avoidBankConflicts??!1}get prefixBlockWorkgroupCount(){return 4*this.workgroupCount}createShaderModules(){this.shaderModules.blockSum=this.device.createShaderModule({label:"radix-sort-block-sum",code:this.blockSumSource}),this.shaderModules.reorder=this.device.createShaderModule({label:"radix-sort-reorder",code:this.reorderSource})}createPipelines(){this.#e();const e=this.#t();this.createResources(),this.#i(e),this.createCheckSortKernels(e);for(let t=0;t<this.bitCount;t+=2){const i=t%4===0,r=this.getPassInData(i),s=this.getPassOutData(i),u=this.createBlockSumPipeline(r,t),a=this.createReorderPipeline(r,s,t);this.pipelines.push(u,a)}}#e(){const e=this.device.createBuffer({label:"radix-sort-prefix-block-sum",size:this.prefixBlockWorkgroupCount*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST}),t=new x({device:this.device,data:e,count:this.prefixBlockWorkgroupCount,workgroupSize:this.workgroupSize,avoidBankConflicts:this.avoidBankConflicts});this.kernels.prefixSum=t,this.buffers.prefixBlockSum=e}#t(){const e=p(this.device,this.workgroupCount),t=this.kernels.prefixSum.getDispatchChain(),i=Math.min(this.count,this.threadsPerWorkgroup*4),r=this.count-i,s=i-1,u=h.findOptimalDispatchChain(this.device,i,this.workgroupSize),a=h.findOptimalDispatchChain(this.device,r,this.workgroupSize),n=[e.x,e.y,1,...u.slice(0,3),...t];return this.dispatchSize=e,this.initialDispatch=n,{initialDispatch:n,dispatchSizesFull:a,checkSortFastCount:i,checkSortFullCount:r,startFull:s}}#i(e){this.checkOrder&&(this.buffers.dispatchSize=_({device:this.device,label:"radix-sort-dispatch-size",data:e.initialDispatch,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.INDIRECT}),this.buffers.originalDispatchSize=_({device:this.device,label:"radix-sort-dispatch-size-original",data:e.initialDispatch,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC}),this.buffers.checkSortFullDispatchSize=_({label:"check-sort-full-dispatch-size",device:this.device,data:e.dispatchSizesFull,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.INDIRECT}),this.buffers.originalCheckSortFullDispatchSize=_({label:"check-sort-full-dispatch-size-original",device:this.device,data:e.dispatchSizesFull,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC}),this.buffers.isSorted=_({label:"is-sorted",device:this.device,data:[0],usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST}))}dispatch(e){this.checkOrder?this.#s(e):this.#r(e)}#r(e){for(let t=0;t<this.bitCount/2;t+=1){const i=this.pipelines[t*2],r=this.pipelines[t*2+1];e.setPipeline(i.pipeline),e.setBindGroup(0,i.bindGroup),e.dispatchWorkgroups(this.dispatchSize.x,this.dispatchSize.y,1),this.kernels.prefixSum.dispatch(e),e.setPipeline(r.pipeline),e.setBindGroup(0,r.bindGroup),e.dispatchWorkgroups(this.dispatchSize.x,this.dispatchSize.y,1)}}#s(e){this.kernels.checkSortReset.dispatch(e);for(let t=0;t<this.bitCount/2;t++){const i=this.pipelines[t*2],r=this.pipelines[t*2+1];t%2==0&&(this.kernels.checkSortFast.dispatch(e,this.buffers.dispatchSize,this.dispatchOffsets.checkSortFast),this.kernels.checkSortFull.dispatch(e,this.buffers.checkSortFullDispatchSize)),e.setPipeline(i.pipeline),e.setBindGroup(0,i.bindGroup),e.dispatchWorkgroupsIndirect(this.buffers.dispatchSize,this.dispatchOffsets.radixSort),this.kernels.prefixSum.dispatch(e,this.buffers.dispatchSize,this.dispatchOffsets.prefixSum),e.setPipeline(r.pipeline),e.setBindGroup(0,r.bindGroup),e.dispatchWorkgroupsIndirect(this.buffers.dispatchSize,this.dispatchOffsets.radixSort)}}}const T=o=>`
${o==="buffer"?`
      @group(0) @binding(0) var<storage, read> input: array<u32>;
      @group(0) @binding(1) var<storage, read_write> local_prefix_sums: array<u32>;
    `:`
      @group(0) @binding(0) var input: texture_storage_2d<rg32uint, read>;
      @group(0) @binding(1) var local_prefix_sums: texture_storage_2d<r32uint, write>;
    `}
@group(0) @binding(2) var<storage, read_write> block_sums: array<u32>;

override WORKGROUP_COUNT: u32;
override THREADS_PER_WORKGROUP: u32;
override WORKGROUP_SIZE_X: u32;
override WORKGROUP_SIZE_Y: u32;
override CURRENT_BIT: u32;
override ELEMENT_COUNT: u32;

var<workgroup> s_prefix_sum: array<u32, 2u * (THREADS_PER_WORKGROUP + 1u)>;

fn getInput(index: u32) -> u32 {
  ${o==="buffer"?"return input[index];":`
        let dimX = textureDimensions(input).r;
        let x = i32(index % dimX);
        let y = i32(index / dimX);
        return textureLoad(input, vec2<i32>(x, y)).x;
      `}
}

fn setLocalPrefixSum(index: u32, val: u32) {
  ${o==="buffer"?"local_prefix_sums[index] = val;":`
        let dimX = textureDimensions(local_prefix_sums).x;
        let x = i32(index % dimX);
        let y = i32(index / dimX);
        textureStore(local_prefix_sums, vec2<i32>(x, y), vec4<u32>(val, 0u, 0u, 0u));
      `}
}

@compute @workgroup_size(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, 1)
fn radix_sort(
  @builtin(workgroup_id) w_id: vec3<u32>,
  @builtin(num_workgroups) w_dim: vec3<u32>,
  @builtin(local_invocation_index) TID: u32, // Local thread ID
) {
  let WORKGROUP_ID = w_id.x + w_id.y * w_dim.x;
  let WID = WORKGROUP_ID * THREADS_PER_WORKGROUP;
  let GID = WID + TID; // Global thread ID

  // Extract 2 bits from the input
  let elm = select(getInput(GID), 0u, GID >= ELEMENT_COUNT);
  let extract_bits: u32 = (elm >> CURRENT_BIT) & 0x3u;

  var bit_prefix_sums = array<u32, 4>(0, 0, 0, 0);

  // If the workgroup is inactive, prevent block_sums buffer update
  var LAST_THREAD: u32 = 0xffffffffu;

  if (WORKGROUP_ID < WORKGROUP_COUNT) {
    // Otherwise store the index of the last active thread in the workgroup
    LAST_THREAD = min(THREADS_PER_WORKGROUP, ELEMENT_COUNT - WID) - 1u;
  }

  // Initialize parameters for double-buffering
  let TPW = THREADS_PER_WORKGROUP + 1u;
  var swapOffset: u32 = 0u;
  var inOffset:  u32 = TID;
  var outOffset: u32 = TID + TPW;

  // 4-way prefix sum
  for (var b: u32 = 0u; b < 4u; b++) {
    // Initialize local prefix with bitmask
    let bitmask = select(0u, 1u, extract_bits == b);
    s_prefix_sum[inOffset + 1u] = bitmask;
    workgroupBarrier();

    var prefix_sum: u32 = 0u;

    // Prefix sum
    for (var offset: u32 = 1u; offset < THREADS_PER_WORKGROUP; offset *= 2u) {
      if (TID >= offset) {
        prefix_sum = s_prefix_sum[inOffset] + s_prefix_sum[inOffset - offset];
      } else {
        prefix_sum = s_prefix_sum[inOffset];
      }

      s_prefix_sum[outOffset] = prefix_sum;

      // Swap buffers
      outOffset = inOffset;
      swapOffset = TPW - swapOffset;
      inOffset = TID + swapOffset;

      workgroupBarrier();
    }

    // Store prefix sum for current bit
    bit_prefix_sums[b] = prefix_sum;

    if (TID == LAST_THREAD) {
      // Store block sum to global memory
      let total_sum: u32 = prefix_sum + bitmask;
      block_sums[b * WORKGROUP_COUNT + WORKGROUP_ID] = total_sum;
    }

    // Swap buffers
    outOffset = inOffset;
    swapOffset = TPW - swapOffset;
    inOffset = TID + swapOffset;
  }

  if (GID < ELEMENT_COUNT) {
    // Store local prefix sum to global memory
    setLocalPrefixSum(GID, bit_prefix_sums[extract_bits]);
  }
}
`,I=`
@group(0) @binding(0) var<storage, read_write> input: array<u32>;
@group(0) @binding(1) var<storage, read_write> local_prefix_sums: array<u32>;
@group(0) @binding(2) var<storage, read_write> block_sums: array<u32>;
@group(0) @binding(3) var<storage, read_write> values: array<u32>;

override WORKGROUP_COUNT: u32;
override THREADS_PER_WORKGROUP: u32;
override WORKGROUP_SIZE_X: u32;
override WORKGROUP_SIZE_Y: u32;
override CURRENT_BIT: u32;
override ELEMENT_COUNT: u32;

var<workgroup> s_prefix_sum: array<u32, 2u * (THREADS_PER_WORKGROUP + 1u)>;
var<workgroup> s_prefix_sum_scan: array<u32, 4>;

@compute @workgroup_size(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, 1)
fn radix_sort(
  @builtin(workgroup_id) w_id: vec3<u32>,
  @builtin(num_workgroups) w_dim: vec3<u32>,
  @builtin(local_invocation_index) TID: u32, // Local thread ID
) {
  let WORKGROUP_ID = w_id.x + w_id.y * w_dim.x;
  let WID = WORKGROUP_ID * THREADS_PER_WORKGROUP;
  let GID = WID + TID; // Global thread ID

  // Extract 2 bits from the input
  var elm: u32 = 0u;
  var val: u32 = 0u;
  if (GID < ELEMENT_COUNT) {
    elm = input[GID];
    val = values[GID];
  }
  let extract_bits: u32 = (elm >> CURRENT_BIT) & 0x3u;

  var bit_prefix_sums = array<u32, 4>(0, 0, 0, 0);

  // If the workgroup is inactive, prevent block_sums buffer update
  var LAST_THREAD: u32 = 0xffffffffu; 

  if (WORKGROUP_ID < WORKGROUP_COUNT) {
    // Otherwise store the index of the last active thread in the workgroup
    LAST_THREAD = min(THREADS_PER_WORKGROUP, ELEMENT_COUNT - WID) - 1u;
  }

  // Initialize parameters for double-buffering
  let TPW = THREADS_PER_WORKGROUP + 1u;
  var swapOffset: u32 = 0u;
  var inOffset:  u32 = TID;
  var outOffset: u32 = TID + TPW;

  // 4-way prefix sum
  for (var b: u32 = 0u; b < 4u; b++) {
    // Initialize local prefix with bitmask
    let bitmask = select(0u, 1u, extract_bits == b);
    s_prefix_sum[inOffset + 1u] = bitmask;
    workgroupBarrier();

    var prefix_sum: u32 = 0u;

    // Prefix sum
    for (var offset: u32 = 1u; offset < THREADS_PER_WORKGROUP; offset *= 2u) {
      if (TID >= offset) {
        prefix_sum = s_prefix_sum[inOffset] + s_prefix_sum[inOffset - offset];
      } else {
        prefix_sum = s_prefix_sum[inOffset];
      }

      s_prefix_sum[outOffset] = prefix_sum;

      // Swap buffers
      outOffset = inOffset;
      swapOffset = TPW - swapOffset;
      inOffset = TID + swapOffset;

      workgroupBarrier();
    }

    // Store prefix sum for current bit
    bit_prefix_sums[b] = prefix_sum;

    if (TID == LAST_THREAD) {
      // Store block sum to global memory
      let total_sum: u32 = prefix_sum + bitmask;
      block_sums[b * WORKGROUP_COUNT + WORKGROUP_ID] = total_sum;
    }

    // Swap buffers
    outOffset = inOffset;
    swapOffset = TPW - swapOffset;
    inOffset = TID + swapOffset;
  }

  let prefix_sum = bit_prefix_sums[extract_bits];   

  // Scan bit prefix sums
  if (TID == LAST_THREAD) {
    var sum: u32 = 0u;
    bit_prefix_sums[extract_bits] += 1u;

    for (var i: u32 = 0u; i < 4u; i++) {
      s_prefix_sum_scan[i] = sum;
      sum += bit_prefix_sums[i];
    }
  }
  workgroupBarrier();

  if (GID < ELEMENT_COUNT) {
    // Compute new position
    let new_pos: u32 = prefix_sum + s_prefix_sum_scan[extract_bits];

    // Shuffle elements locally
    input[WID + new_pos] = elm;
    values[WID + new_pos] = val;
    local_prefix_sums[WID + new_pos] = prefix_sum;
  }
}
`,g=o=>`
${o==="buffer"?`
      @group(0) @binding(0) var<storage, read> inputKeys: array<u32>;
      @group(0) @binding(1) var<storage, read_write> outputKeys: array<u32>;
      @group(0) @binding(2) var<storage, read> local_prefix_sum: array<u32>;
    `:`
      @group(0) @binding(0) var input: texture_storage_2d<rg32uint, read>;
      @group(0) @binding(1) var output: texture_storage_2d<rg32uint, write>;
      @group(0) @binding(2) var local_prefix_sum: texture_storage_2d<r32uint, read_write>;
    `}
@group(0) @binding(3) var<storage, read> prefix_block_sum: array<u32>;
${o==="buffer"?`
      @group(0) @binding(4) var<storage, read> inputValues: array<u32>;
      @group(0) @binding(5) var<storage, read_write> outputValues: array<u32>;
    `:""}

override WORKGROUP_COUNT: u32;
override THREADS_PER_WORKGROUP: u32;
override WORKGROUP_SIZE_X: u32;
override WORKGROUP_SIZE_Y: u32;
override CURRENT_BIT: u32;
override ELEMENT_COUNT: u32;

fn getInput(index: u32) -> vec2<u32> {
  ${o==="buffer"?`
        let result: vec2<u32> = vec2<u32>(
          inputKeys[index],
          inputValues[index]
        );
        return result;
      `:`
        let dimX = textureDimensions(input).x;
        let x = i32(index % dimX);
        let y = i32(index / dimY);
        return textureLoad(input, vec2<i32>(x, y)).xy;
      `}
}

fn setOutput(index: u32, key: u32, val: u32) {
  ${o==="buffer"?`
        outputKeys[index] = key;
        outputValues[index] = val;
      `:`
        let dimX = textureDimensions(output).x;
        let x = i32(index % dimX);
        let y = i32(index / dimX);
        textureStore(output, vec2<i32>(x, y), vec4<u32>(key, val, 0u, 0u));
      `}
}

fn getLocalPrefixSum(index: u32) -> u32 {
  ${o==="buffer"?"return local_prefix_sum[index];":`
        let dimX = textureDimensions(local_prefix_sum).x;
        let x = i32(index % dimX);
        let y = i32(index / dimX);
        return textureLoad(local_prefix_sum, vec2<i32>(x, y)).x;
      `}
}

@compute @workgroup_size(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, 1)
fn radix_sort_reorder(
  @builtin(workgroup_id) w_id: vec3<u32>,
  @builtin(num_workgroups) w_dim: vec3<u32>,
  @builtin(local_invocation_index) TID: u32, // Local thread ID
) {
  let WORKGROUP_ID = w_id.x + w_id.y * w_dim.x;
  let WID = WORKGROUP_ID * THREADS_PER_WORKGROUP;
  let GID = WID + TID; // Global thread ID

  if (GID >= ELEMENT_COUNT) { return; }

  let input = getInput(GID);

  let local_prefix = getLocalPrefixSum(GID);

  // Calculate new position
  let extract_bits = (input.x >> CURRENT_BIT) & 0x3u;
  let pid = extract_bits * WORKGROUP_COUNT + WORKGROUP_ID;
  let sorted_position = prefix_block_sum[pid] + local_prefix;

  setOutput(sorted_position, input.x, input.y);
}
`,G=(o=!1,e=!1,t="full",i)=>`
${i==="buffer"?`
      @group(0) @binding(0) var<storage, read> input: array<u32>;
      @group(0) @binding(1) var<storage, read_write> output: array<u32>;
    `:`
      @group(0) @binding(0) var input: texture_storage_2d<rg32uint, read>;
      @group(0) @binding(1) var output: texture_storage_2d<r32uint, write>;
    `}
@group(0) @binding(2) var<storage, read> original: array<u32>;
@group(0) @binding(3) var<storage, read_write> is_sorted: u32;

override WORKGROUP_SIZE_X: u32;
override WORKGROUP_SIZE_Y: u32;
override THREADS_PER_WORKGROUP: u32;
override ELEMENT_COUNT: u32;
override START_ELEMENT: u32;

var<workgroup> s_data: array<u32, THREADS_PER_WORKGROUP>;

fn getInput(index: u32) -> u32 {
  ${i==="buffer"?"return input[index];":`
        let dimX = textureDimensions(input).r;
        let x = i32(index % dimX);
        let y = i32(index / dimX);
        return textureLoad(input, vec2<i32>(x, y)).x;
      `}
}

fn setOutput(index: u32, data: u32) {
  ${i==="buffer"?"output[index] = data;":`
        let dimX = textureDimensions(output).x;
        let x = i32(index % dimX);
        let y = i32(index / dimX);
        textureStore(output, vec2<i32>(x, y), vec4<u32>(data, 0u, 0u, 0u));
      `}
}

// Reset dispatch buffer and is_sorted flag
@compute @workgroup_size(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, 1)
fn reset(
  @builtin(workgroup_id) w_id: vec3<u32>,
  @builtin(num_workgroups) w_dim: vec3<u32>,
  @builtin(local_invocation_index) TID: u32, // Local thread ID
) {
  if (TID >= ELEMENT_COUNT) {
    return;
  }

  if (TID == 0) {
    is_sorted = 0u;
  }

  let ELM_ID = TID * 3;

  setOutput(ELM_ID, original[ELM_ID]);
}

@compute @workgroup_size(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, 1)
fn check_sort(
  @builtin(workgroup_id) w_id: vec3<u32>,
  @builtin(num_workgroups) w_dim: vec3<u32>,
  @builtin(local_invocation_index) TID: u32, // Local thread ID
) {
  let WORKGROUP_ID = w_id.x + w_id.y * w_dim.x;
  let WID = WORKGROUP_ID * THREADS_PER_WORKGROUP + START_ELEMENT;
  let GID = TID + WID; // Global thread ID

  // Load data into shared memory
  ${o?y:"s_data[TID] = select(0u, getInput(GID), GID < ELEMENT_COUNT);"}

  // Perform parallel reduction
  for (var d = 1u; d < THREADS_PER_WORKGROUP; d *= 2u) {
    workgroupBarrier();
    if (TID % (2u * d) == 0u) {
      s_data[TID] += s_data[TID + d];
    }
  }
  workgroupBarrier();

  // Write reduction result
  ${e?W(t,i):w}
}`,w=`
  if (TID == 0) {
    setOutput(WORKGROUP_ID, s_data[0]);
  }
`,y=`
  let LAST_THREAD = min(THREADS_PER_WORKGROUP, ELEMENT_COUNT - WID) - 1;

  // Load current element into shared memory
  // Also load next element for comparison
  let elm = select(0u, getInput(GID), GID < ELEMENT_COUNT);
  let next = select(0u, getInput(GID + 1), GID < ELEMENT_COUNT-1);
  s_data[TID] = elm;
  workgroupBarrier();

  s_data[TID] = select(0u, 1u, GID < ELEMENT_COUNT-1 && elm > next);
`,W=(o,e="buffer")=>`
  ${e==="buffer"?"let fullDispatchLength = arrayLength(&output);":`
        let dim = textureDimensions(output);
        let fullDispatchLength = dim.x * dim.y;
      `}
  let dispatchIndex = TID * 3;

  if (dispatchIndex >= fullDispatchLength) {
    return;
  }

  ${o=="full"?K:C}
`,C=`
  setOutput(dispatchIndex, select(0, original[dispatchIndex], s_data[0] == 0 && is_sorted == 0u));
`,K=`
  if (TID == 0 && s_data[0] == 0) {
    is_sorted = 1u;
  }

  setOutput(dispatchIndex, select(0, original[dispatchIndex], s_data[0] != 0));
`;class b extends h{outputs=[];constructor(e){super(e),this.buffers.data=e.data.keys,this.createPassesRecursive(e.data,this.count)}createPassesRecursive(e,t,i=0){const r=Math.ceil(t/this.threadsPerWorkgroup),s=!i,u=r<=1,a=`check-sort-${this.mode}-${i}`,n=u?this.buffers.result:this.device.createBuffer({label:a,size:r*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST}),d=this.device.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},{binding:1,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},...u?[{binding:2,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},{binding:3,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}}]:[]]}),f=this.device.createBindGroup({layout:d,entries:[{binding:0,resource:{buffer:e.keys}},{binding:1,resource:{buffer:n}},...u?[{binding:2,resource:{buffer:this.buffers.original}},{binding:3,resource:{buffer:this.buffers.isSorted}}]:[]]}),S=this.device.createPipelineLayout({bindGroupLayouts:[d]}),R=s?this.start+t:t,P=s?this.start:0,m=this.device.createComputePipeline({layout:S,compute:{module:this.device.createShaderModule({label:a,code:G(s,u,this.mode,"buffer")}),entryPoint:this.mode=="reset"?"reset":"check_sort",constants:{ELEMENT_COUNT:R,WORKGROUP_SIZE_X:this.workgroupSize.x,WORKGROUP_SIZE_Y:this.workgroupSize.y,...this.mode!=="reset"&&{THREADS_PER_WORKGROUP:this.threadsPerWorkgroup,START_ELEMENT:P}}}});this.outputs.push(n),this.pipelines.push({pipeline:m,bindGroup:f}),u||this.createPassesRecursive({keys:n},r,i+1)}}class M extends U{constructor(e){super(e),this.localShuffle=e.localShuffle??!1,this.buffers.keys=e.data.keys,e.data.values&&(this.buffers.values=e.data.values),this.createShaderModules(),this.createPipelines()}get hasValues(){return!!this.data.values}get blockSumSource(){const e=this.localShuffle?I:T("buffer");return this.hasValues?e:E(e)}get reorderSource(){return this.hasValues?g("buffer"):E(g("buffer"))}createResources(){this.buffers.tmpKeys=this.device.createBuffer({label:"radix-sort-tmp-keys",size:this.count*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST}),this.hasValues&&(this.buffers.tmpValues=this.device.createBuffer({label:"radix-sort-tmp-values",size:this.count*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST})),this.buffers.localPrefixSum=this.device.createBuffer({label:"radix-sort-local-prefix-sum",size:this.count*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST})}getPassInData(e){return{keys:e?this.buffers.keys:this.buffers.tmpKeys,values:e?this.buffers.values:this.buffers.tmpValues}}getPassOutData(e){return{keys:e?this.buffers.tmpKeys:this.buffers.keys,values:e?this.buffers.tmpValues:this.buffers.values}}createBlockSumPipeline(e,t){const i=this.device.createBindGroupLayout({label:"radix-sort-block-sum",entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:this.localShuffle?"storage":"read-only-storage"}},{binding:1,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},{binding:2,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},...this.localShuffle&&this.hasValues?[{binding:3,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}}]:[]]}),r=this.device.createBindGroup({layout:i,entries:[{binding:0,resource:{buffer:e.keys}},{binding:1,resource:{buffer:this.buffers.localPrefixSum}},{binding:2,resource:{buffer:this.buffers.prefixBlockSum}},...this.localShuffle&&this.hasValues?[{binding:3,resource:{buffer:e.values}}]:[]]}),s=this.device.createPipelineLayout({bindGroupLayouts:[i]});return{pipeline:this.device.createComputePipeline({label:"radix-sort-block-sum",layout:s,compute:{module:this.shaderModules.blockSum,entryPoint:"radix_sort",constants:{WORKGROUP_SIZE_X:this.workgroupSize.x,WORKGROUP_SIZE_Y:this.workgroupSize.y,WORKGROUP_COUNT:this.workgroupCount,THREADS_PER_WORKGROUP:this.threadsPerWorkgroup,ELEMENT_COUNT:this.count,CURRENT_BIT:t}}}),bindGroup:r}}createCheckSortKernels(e){if(!this.checkOrder)return;const{checkSortFastCount:t,checkSortFullCount:i,startFull:r}=e;this.kernels.checkSortFull=new b({mode:"full",device:this.device,data:this.data,result:this.buffers.dispatchSize,original:this.buffers.originalDispatchSize,isSorted:this.buffers.isSorted,count:i,start:r,workgroupSize:this.workgroupSize}),this.kernels.checkSortFast=new b({mode:"fast",device:this.device,data:this.data,result:this.buffers.checkSortFullDispatchSize,original:this.buffers.originalCheckSortFullDispatchSize,isSorted:this.buffers.isSorted,count:t,workgroupSize:this.workgroupSize});const s=this.initialDispatch.length/3;if(this.kernels.checkSortFast.threadsPerWorkgroup<this.kernels.checkSortFull.pipelines.length||this.kernels.checkSortFull.threadsPerWorkgroup<s){console.warn("Warning: workgroup size is too small to enable check sort optimization, disabling..."),this.checkOrder=!1;return}this.kernels.checkSortReset=new b({mode:"reset",device:this.device,data:this.data,original:this.buffers.originalDispatchSize,result:this.buffers.dispatchSize,isSorted:this.buffers.isSorted,count:s,workgroupSize:p(this.device,s)})}createReorderPipeline(e,t,i){const r=this.device.createBindGroupLayout({label:"radix-sort-reorder",entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},{binding:1,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},{binding:2,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},{binding:3,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},...this.hasValues?[{binding:4,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},{binding:5,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}}]:[]]}),s=this.device.createBindGroup({layout:r,entries:[{binding:0,resource:{buffer:e.keys}},{binding:1,resource:{buffer:t.keys}},{binding:2,resource:{buffer:this.buffers.localPrefixSum}},{binding:3,resource:{buffer:this.buffers.prefixBlockSum}},...this.hasValues?[{binding:4,resource:{buffer:e.values}},{binding:5,resource:{buffer:t.values}}]:[]]}),u=this.device.createPipelineLayout({bindGroupLayouts:[r]});return{pipeline:this.device.createComputePipeline({label:"radix-sort-reorder",layout:u,compute:{module:this.shaderModules.reorder,entryPoint:"radix_sort_reorder",constants:{WORKGROUP_SIZE_X:this.workgroupSize.x,WORKGROUP_SIZE_Y:this.workgroupSize.y,WORKGROUP_COUNT:this.workgroupCount,THREADS_PER_WORKGROUP:this.threadsPerWorkgroup,ELEMENT_COUNT:this.count,CURRENT_BIT:i}}}),bindGroup:s}}}class O extends h{textures={};outputs=[];constructor(e){super(e),this.textures.read=e.data.texture,this.createPassesRecursive(e.data,this.count)}createPassesRecursive(e,t,i=0){const r=Math.ceil(t/this.threadsPerWorkgroup),s=!i,u=r<=1,a=`check-sort-${this.mode}-${i}`,n=u?this.buffers.result:this.device.createBuffer({label:a,size:r*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST}),d=D(this.device,n),f=this.device.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,storageTexture:{access:"read-only",format:"rg32uint",viewDimension:"2d"}},{binding:1,visibility:GPUShaderStage.COMPUTE,storageTexture:{access:"read-write",format:"r32uint",viewDimension:"2d"}},...u?[{binding:2,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},{binding:3,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}}]:[]]}),S=this.device.createBindGroup({layout:f,entries:[{binding:0,resource:e.texture.createView()},{binding:1,resource:d.createView()},...u?[{binding:2,resource:{buffer:this.buffers.original}},{binding:3,resource:{buffer:this.buffers.isSorted}}]:[]]}),R=this.device.createPipelineLayout({bindGroupLayouts:[f]}),P=s?this.start+t:t,m=s?this.start:0,N=this.device.createComputePipeline({layout:R,compute:{module:this.device.createShaderModule({label:a,code:G(s,u,this.mode,"texture")}),entryPoint:this.mode=="reset"?"reset":"check_sort",constants:{ELEMENT_COUNT:P,WORKGROUP_SIZE_X:this.workgroupSize.x,WORKGROUP_SIZE_Y:this.workgroupSize.y,...this.mode!=="reset"&&{THREADS_PER_WORKGROUP:this.threadsPerWorkgroup,START_ELEMENT:m}}}});this.outputs.push(d),this.pipelines.push({pipeline:N,bindGroup:S}),u||this.createPassesRecursive({texture:d},r,i+1)}}class L extends U{textures={};constructor(e){super(e),this.textures.read=e.data.texture,this.createShaderModules(),this.createPipelines()}get hasValues(){return!0}get blockSumSource(){return T("texture")}get reorderSource(){return g("texture")}createResources(){this.textures.write=this.device.createTexture({size:{width:this.textures.read.width,height:this.textures.read.height},format:this.textures.read.format,usage:GPUTextureUsage.STORAGE_BINDING|GPUTextureUsage.COPY_DST|GPUTextureUsage.COPY_SRC}),this.textures.localPrefixSum=this.device.createTexture({size:{width:this.textures.read.width,height:this.textures.read.height},format:"r32uint",usage:GPUTextureUsage.STORAGE_BINDING|GPUTextureUsage.COPY_DST|GPUTextureUsage.COPY_SRC})}getPassInData(e){return{texture:e?this.textures.read:this.textures.write}}getPassOutData(e){return{texture:e?this.textures.write:this.textures.read}}createBlockSumPipeline(e,t){const i=this.device.createBindGroupLayout({label:"radix-sort-block-sum",entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,storageTexture:{access:"read-only",format:"rg32uint",viewDimension:"2d"}},{binding:1,visibility:GPUShaderStage.COMPUTE,storageTexture:{access:"write-only",format:"r32uint",viewDimension:"2d"}},{binding:2,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}}]}),r=this.device.createBindGroup({layout:i,entries:[{binding:0,resource:e.texture.createView()},{binding:1,resource:this.textures.localPrefixSum.createView()},{binding:2,resource:{buffer:this.buffers.prefixBlockSum}}]}),s=this.device.createPipelineLayout({bindGroupLayouts:[i]});return{pipeline:this.device.createComputePipeline({label:"radix-sort-block-sum",layout:s,compute:{module:this.shaderModules.blockSum,entryPoint:"radix_sort",constants:{WORKGROUP_SIZE_X:this.workgroupSize.x,WORKGROUP_SIZE_Y:this.workgroupSize.y,WORKGROUP_COUNT:this.workgroupCount,THREADS_PER_WORKGROUP:this.threadsPerWorkgroup,ELEMENT_COUNT:this.count,CURRENT_BIT:t}}}),bindGroup:r}}createCheckSortKernels(e){if(!this.checkOrder)return;const{checkSortFastCount:t,checkSortFullCount:i,startFull:r}=e;this.kernels.checkSortFull=new O({mode:"full",device:this.device,data:this.data,result:this.buffers.dispatchSize,original:this.buffers.originalDispatchSize,isSorted:this.buffers.isSorted,count:i,start:r,workgroupSize:this.workgroupSize}),this.kernels.checkSortFast=new O({mode:"fast",device:this.device,data:this.data,result:this.buffers.checkSortFullDispatchSize,original:this.buffers.originalCheckSortFullDispatchSize,isSorted:this.buffers.isSorted,count:t,workgroupSize:this.workgroupSize});const s=this.initialDispatch.length/3;if(this.kernels.checkSortFast.threadsPerWorkgroup<this.kernels.checkSortFull.pipelines.length||this.kernels.checkSortFull.threadsPerWorkgroup<s){console.warn("Warning: workgroup size is too small to enable check sort optimization, disabling..."),this.checkOrder=!1;return}this.kernels.checkSortReset=new O({mode:"reset",device:this.device,data:this.data,original:this.buffers.originalDispatchSize,result:this.buffers.dispatchSize,isSorted:this.buffers.isSorted,count:s,workgroupSize:p(this.device,s)})}createReorderPipeline(e,t,i){const r=this.device.createBindGroupLayout({label:"radix-sort-reorder",entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,storageTexture:{access:"read-only",format:"rg32uint",viewDimension:"2d"}},{binding:1,visibility:GPUShaderStage.COMPUTE,storageTexture:{access:"write-only",format:"rg32uint",viewDimension:"2d"}},{binding:2,visibility:GPUShaderStage.COMPUTE,storageTexture:{access:"read-write",format:"r32uint",viewDimension:"2d"}},{binding:3,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}}]}),s=this.device.createBindGroup({layout:r,entries:[{binding:0,resource:e.texture.createView()},{binding:1,resource:t.texture.createView()},{binding:2,resource:this.textures.localPrefixSum.createView()},{binding:3,resource:{buffer:this.buffers.prefixBlockSum}}]}),u=this.device.createPipelineLayout({bindGroupLayouts:[r]});return{pipeline:this.device.createComputePipeline({label:"radix-sort-reorder",layout:u,compute:{module:this.shaderModules.reorder,entryPoint:"radix_sort_reorder",constants:{WORKGROUP_SIZE_X:this.workgroupSize.x,WORKGROUP_SIZE_Y:this.workgroupSize.y,WORKGROUP_COUNT:this.workgroupCount,THREADS_PER_WORKGROUP:this.threadsPerWorkgroup,ELEMENT_COUNT:this.count,CURRENT_BIT:i}}}),bindGroup:s}}}l.PrefixSumKernel=x,l.RadixSortBufferKernel=M,l.RadixSortTextureKernel=L,Object.defineProperty(l,Symbol.toStringTag,{value:"Module"})});
//# sourceMappingURL=radix-sort.umd.js.map
