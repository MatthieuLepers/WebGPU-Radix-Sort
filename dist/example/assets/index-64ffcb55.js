var Z=Object.defineProperty;var X=(u,t,e)=>t in u?Z(u,t,{enumerable:!0,configurable:!0,writable:!0,value:e}):u[t]=e;var d=(u,t,e)=>(X(u,typeof t!="symbol"?t+"":t,e),e),$=(u,t,e)=>{if(!t.has(u))throw TypeError("Cannot "+e)};var b=(u,t,e)=>{if(t.has(u))throw TypeError("Cannot add the same private member more than once");t instanceof WeakSet?t.add(u):t.set(u,e)};var O=(u,t,e)=>($(u,t,"access private method"),e);(function(){const t=document.createElement("link").relList;if(t&&t.supports&&t.supports("modulepreload"))return;for(const i of document.querySelectorAll('link[rel="modulepreload"]'))r(i);new MutationObserver(i=>{for(const s of i)if(s.type==="childList")for(const a of s.addedNodes)a.tagName==="LINK"&&a.rel==="modulepreload"&&r(a)}).observe(document,{childList:!0,subtree:!0});function e(i){const s={};return i.integrity&&(s.integrity=i.integrity),i.referrerPolicy&&(s.referrerPolicy=i.referrerPolicy),i.crossOrigin==="use-credentials"?s.credentials="include":i.crossOrigin==="anonymous"?s.credentials="omit":s.credentials="same-origin",s}function r(i){if(i.ep)return;i.ep=!0;const s=e(i);fetch(i.href,s)}})();class C{constructor(t){d(this,"options");d(this,"workgroupSize",{x:16,y:16});d(this,"pipelines",[]);d(this,"shaderModules",{});this.options=t,Object.keys(t).forEach(e=>{Object.defineProperty(this,e,{get:()=>this.options[e],set:r=>{this.options[e]=r}})})}get workgroupCount(){return Math.ceil(this.count/this.threadsPerWorkgroup)}get threadsPerWorkgroup(){return this.workgroupSize.x*this.workgroupSize.y}get itemsPerWorkgroup(){return 2*this.threadsPerWorkgroup}}function I(u,t){const e={x:t,y:1};if(t>u.limits.maxComputeWorkgroupsPerDimension){const r=Math.floor(Math.sqrt(t)),i=Math.ceil(t/r);e.x=r,e.y=i}return e}function P({device:u,label:t,data:e,usage:r=0}){const i=u.createBuffer({label:t,usage:r,size:e.length*4,mappedAtCreation:!0});return new Uint32Array(i.getMappedRange()).set(e),i.unmap(),i}const M=u=>u.split(`
`).filter(t=>!t.toLowerCase().includes("values")).join(`
`),F=`
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
`,V=`
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
`;class q extends C{constructor({device:t,count:e,workgroupSize:r={x:16,y:16},data:i,avoidBankConflicts:s=!1}){if(super({device:t,count:e,workgroupSize:r}),Math.log2(this.threadsPerWorkgroup)%1!==0)throw new Error(`workgroupSize.x * workgroupSize.y must be a power of two. (current: ${this.threadsPerWorkgroup})`);this.shaderModules.prefixSum=this.device.createShaderModule({label:"prefix-sum",code:s?V:F}),this.createPassRecursive(i,e)}createPassRecursive(t,e){const r=Math.ceil(e/this.itemsPerWorkgroup),i=I(this.device,r),s=this.device.createBuffer({label:"prefix-sum-block-sum",size:r*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST}),a=this.device.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},{binding:1,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}}]}),o=this.device.createBindGroup({label:"prefix-sum-bind-group",layout:a,entries:[{binding:0,resource:{buffer:t}},{binding:1,resource:{buffer:s}}]}),l=this.device.createPipelineLayout({bindGroupLayouts:[a]}),f=this.device.createComputePipeline({label:"prefix-sum-scan-pipeline",layout:l,compute:{module:this.shaderModules.prefixSum,entryPoint:"reduce_downsweep",constants:{WORKGROUP_SIZE_X:this.workgroupSize.x,WORKGROUP_SIZE_Y:this.workgroupSize.y,THREADS_PER_WORKGROUP:this.threadsPerWorkgroup,ITEMS_PER_WORKGROUP:this.itemsPerWorkgroup,ELEMENT_COUNT:e}}});if(this.pipelines.push({pipeline:f,bindGroup:o,dispatchSize:i}),r>1){this.createPassRecursive(s,r);const p=this.device.createComputePipeline({label:"prefix-sum-add-block-pipeline",layout:l,compute:{module:this.shaderModules.prefixSum,entryPoint:"add_block_sums",constants:{WORKGROUP_SIZE_X:this.workgroupSize.x,WORKGROUP_SIZE_Y:this.workgroupSize.y,THREADS_PER_WORKGROUP:this.threadsPerWorkgroup,ELEMENT_COUNT:e}}});this.pipelines.push({pipeline:p,bindGroup:o,dispatchSize:i})}}getDispatchChain(){return this.pipelines.flatMap(t=>[t.dispatchSize.x,t.dispatchSize.y,1])}dispatch(t,e,r=0){this.pipelines.forEach(({pipeline:i,bindGroup:s,dispatchSize:a},o)=>{t.setPipeline(i),t.setBindGroup(0,s),e?t.dispatchWorkgroupsIndirect(e,r+o*3*4):t.dispatchWorkgroups(a.x,a.y,1)})}}class w extends C{constructor(e){super(e);d(this,"buffers",{});this.start=e.start??0,this.mode=e.mode??"full",this.buffers.result=e.result,this.buffers.original=e.original,this.buffers.isSorted=e.isSorted}static findOptimalDispatchChain(e,r,i){const s=i.x*i.y,a=[];do{const o=Math.ceil(r/s),l=I(e,o);a.push(l.x,l.y,1),r=o}while(r>1);return a}dispatch(e,r,i=0){this.pipelines.forEach(({pipeline:s,bindGroup:a},o)=>{const l=this.mode!=="reset"&&(this.mode==="full"||o<this.pipelines.length-1);e.setPipeline(s),e.setBindGroup(0,a),l&&r?e.dispatchWorkgroupsIndirect(r,i+o*3*4):e.dispatchWorkgroups(1,1,1)})}}var v,B,x,N,T,z,U,A,k,H;class Q extends C{constructor(e){super(e);b(this,v);b(this,x);b(this,T);b(this,U);b(this,k);d(this,"buffers",{});d(this,"kernels",{});d(this,"dispatchSize",{x:1,y:1});d(this,"dispatchOffsets",{radixSort:0,checkSortFast:3*4,prefixSum:6*4});d(this,"initialDispatch",[]);this.bitCount=e.bitCount??32,this.checkOrder=e.checkOrder??!1,this.avoidBankConflicts=e.avoidBankConflicts??!1}get prefixBlockWorkgroupCount(){return 4*this.workgroupCount}createShaderModules(){this.shaderModules.blockSum=this.device.createShaderModule({label:"radix-sort-block-sum",code:this.blockSumSource}),this.shaderModules.reorder=this.device.createShaderModule({label:"radix-sort-reorder",code:this.reorderSource})}createPipelines(){O(this,v,B).call(this);const e=O(this,x,N).call(this);this.createResources(),O(this,T,z).call(this,e),this.createCheckSortKernels(e);for(let r=0;r<this.bitCount;r+=2){const i=r%4===0,s=this.getPassInData(i),a=this.getPassOutData(i),o=this.createBlockSumPipeline(s,r),l=this.createReorderPipeline(s,a,r);this.pipelines.push(o,l)}}dispatch(e){this.checkOrder?O(this,k,H).call(this,e):O(this,U,A).call(this,e)}}v=new WeakSet,B=function(){const e=this.device.createBuffer({label:"radix-sort-prefix-block-sum",size:this.prefixBlockWorkgroupCount*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST}),r=new q({device:this.device,data:e,count:this.prefixBlockWorkgroupCount,workgroupSize:this.workgroupSize,avoidBankConflicts:this.avoidBankConflicts});this.kernels.prefixSum=r,this.buffers.prefixBlockSum=e},x=new WeakSet,N=function(){const e=I(this.device,this.workgroupCount),r=this.kernels.prefixSum.getDispatchChain(),i=Math.min(this.count,this.threadsPerWorkgroup*4),s=this.count-i,a=i-1,o=w.findOptimalDispatchChain(this.device,i,this.workgroupSize),l=w.findOptimalDispatchChain(this.device,s,this.workgroupSize),f=[e.x,e.y,1,...o.slice(0,3),...r];return this.dispatchSize=e,this.initialDispatch=f,{initialDispatch:f,dispatchSizesFull:l,checkSortFastCount:i,checkSortFullCount:s,startFull:a}},T=new WeakSet,z=function(e){this.checkOrder&&(this.buffers.dispatchSize=P({device:this.device,label:"radix-sort-dispatch-size",data:e.initialDispatch,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.INDIRECT}),this.buffers.originalDispatchSize=P({device:this.device,label:"radix-sort-dispatch-size-original",data:e.initialDispatch,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC}),this.buffers.checkSortFullDispatchSize=P({label:"check-sort-full-dispatch-size",device:this.device,data:e.dispatchSizesFull,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.INDIRECT}),this.buffers.originalCheckSortFullDispatchSize=P({label:"check-sort-full-dispatch-size-original",device:this.device,data:e.dispatchSizesFull,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC}),this.buffers.isSorted=P({label:"is-sorted",device:this.device,data:[0],usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST}))},U=new WeakSet,A=function(e){for(let r=0;r<this.bitCount/2;r+=1){const i=this.pipelines[r*2],s=this.pipelines[r*2+1];e.setPipeline(i.pipeline),e.setBindGroup(0,i.bindGroup),e.dispatchWorkgroups(this.dispatchSize.x,this.dispatchSize.y,1),this.kernels.prefixSum.dispatch(e),e.setPipeline(s.pipeline),e.setBindGroup(0,s.bindGroup),e.dispatchWorkgroups(this.dispatchSize.x,this.dispatchSize.y,1)}},k=new WeakSet,H=function(e){this.kernels.checkSortReset.dispatch(e);for(let r=0;r<this.bitCount/2;r++){const i=this.pipelines[r*2],s=this.pipelines[r*2+1];r%2==0&&(this.kernels.checkSortFast.dispatch(e,this.buffers.dispatchSize,this.dispatchOffsets.checkSortFast),this.kernels.checkSortFull.dispatch(e,this.buffers.checkSortFullDispatchSize)),e.setPipeline(i.pipeline),e.setBindGroup(0,i.bindGroup),e.dispatchWorkgroupsIndirect(this.buffers.dispatchSize,this.dispatchOffsets.radixSort),this.kernels.prefixSum.dispatch(e,this.buffers.dispatchSize,this.dispatchOffsets.prefixSum),e.setPipeline(s.pipeline),e.setBindGroup(0,s.bindGroup),e.dispatchWorkgroupsIndirect(this.buffers.dispatchSize,this.dispatchOffsets.radixSort)}};const j=u=>`
${u==="buffer"?`
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
  ${u==="buffer"?"return input[index];":`
        let dimX = textureDimensions(input).r;
        let x = i32(index % dimX);
        let y = i32(index / dimX);
        return textureLoad(input, vec2<i32>(x, y)).x;
      `}
}

fn setLocalPrefixSum(index: u32, val: u32) {
  ${u==="buffer"?"local_prefix_sums[index] = val;":`
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
`,J=`
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
`,L=u=>`
${u==="buffer"?`
      @group(0) @binding(0) var<storage, read> inputKeys: array<u32>;
      @group(0) @binding(1) var<storage, read_write> outputKeys: array<u32>;
      @group(0) @binding(2) var<storage, read> local_prefix_sum: array<u32>;
    `:`
      @group(0) @binding(0) var input: texture_storage_2d<rg32uint, read>;
      @group(0) @binding(1) var output: texture_storage_2d<rg32uint, write>;
      @group(0) @binding(2) var local_prefix_sum: texture_storage_2d<r32uint, read_write>;
    `}
@group(0) @binding(3) var<storage, read> prefix_block_sum: array<u32>;
${u==="buffer"?`
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
  ${u==="buffer"?`
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
  ${u==="buffer"?`
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
  ${u==="buffer"?"return local_prefix_sum[index];":`
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
`,ee=(u=!1,t=!1,e="full",r)=>`
${r==="buffer"?`
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
  ${r==="buffer"?"return input[index];":`
        let dimX = textureDimensions(input).r;
        let x = i32(index % dimX);
        let y = i32(index / dimX);
        return textureLoad(input, vec2<i32>(x, y)).x;
      `}
}

fn setOutput(index: u32, data: u32) {
  ${r==="buffer"?"output[index] = data;":`
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
  ${u?ie:"s_data[TID] = select(0u, getInput(GID), GID < ELEMENT_COUNT);"}

  // Perform parallel reduction
  for (var d = 1u; d < THREADS_PER_WORKGROUP; d *= 2u) {
    workgroupBarrier();
    if (TID % (2u * d) == 0u) {
      s_data[TID] += s_data[TID + d];
    }
  }
  workgroupBarrier();

  // Write reduction result
  ${t?re(e,r):te}
}`,te=`
  if (TID == 0) {
    setOutput(WORKGROUP_ID, s_data[0]);
  }
`,ie=`
  let LAST_THREAD = min(THREADS_PER_WORKGROUP, ELEMENT_COUNT - WID) - 1;

  // Load current element into shared memory
  // Also load next element for comparison
  let elm = select(0u, getInput(GID), GID < ELEMENT_COUNT);
  let next = select(0u, getInput(GID + 1), GID < ELEMENT_COUNT-1);
  s_data[TID] = elm;
  workgroupBarrier();

  s_data[TID] = select(0u, 1u, GID < ELEMENT_COUNT-1 && elm > next);
`,re=(u,t="buffer")=>`
  ${t==="buffer"?"let fullDispatchLength = arrayLength(&output);":`
        let dim = textureDimensions(output);
        let fullDispatchLength = dim.x * dim.y;
      `}
  let dispatchIndex = TID * 3;

  if (dispatchIndex >= fullDispatchLength) {
    return;
  }

  ${u=="full"?ue:se}
`,se=`
  setOutput(dispatchIndex, select(0, original[dispatchIndex], s_data[0] == 0 && is_sorted == 0u));
`,ue=`
  if (TID == 0 && s_data[0] == 0) {
    is_sorted = 1u;
  }

  setOutput(dispatchIndex, select(0, original[dispatchIndex], s_data[0] != 0));
`;class G extends w{constructor(e){super(e);d(this,"outputs",[]);this.buffers.data=e.data.keys,this.createPassesRecursive(e.data,this.count)}createPassesRecursive(e,r,i=0){const s=Math.ceil(r/this.threadsPerWorkgroup),a=!i,o=s<=1,l=`check-sort-${this.mode}-${i}`,f=o?this.buffers.result:this.device.createBuffer({label:l,size:s*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST}),p=this.device.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},{binding:1,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},...o?[{binding:2,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},{binding:3,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}}]:[]]}),m=this.device.createBindGroup({layout:p,entries:[{binding:0,resource:{buffer:e.keys}},{binding:1,resource:{buffer:f}},...o?[{binding:2,resource:{buffer:this.buffers.original}},{binding:3,resource:{buffer:this.buffers.isSorted}}]:[]]}),g=this.device.createPipelineLayout({bindGroupLayouts:[p]}),_=a?this.start+r:r,R=a?this.start:0,E=this.device.createComputePipeline({layout:g,compute:{module:this.device.createShaderModule({label:l,code:ee(a,o,this.mode,"buffer")}),entryPoint:this.mode=="reset"?"reset":"check_sort",constants:{ELEMENT_COUNT:_,WORKGROUP_SIZE_X:this.workgroupSize.x,WORKGROUP_SIZE_Y:this.workgroupSize.y,...this.mode!=="reset"&&{THREADS_PER_WORKGROUP:this.threadsPerWorkgroup,START_ELEMENT:R}}}});this.outputs.push(f),this.pipelines.push({pipeline:E,bindGroup:m}),o||this.createPassesRecursive({keys:f},s,i+1)}}class oe extends Q{constructor(t){super(t),this.localShuffle=t.localShuffle??!1,this.buffers.keys=t.data.keys,t.data.values&&(this.buffers.values=t.data.values),this.createShaderModules(),this.createPipelines()}get hasValues(){return!!this.data.values}get blockSumSource(){const t=this.localShuffle?J:j("buffer");return this.hasValues?t:M(t)}get reorderSource(){return this.hasValues?L("buffer"):M(L("buffer"))}createResources(){this.buffers.tmpKeys=this.device.createBuffer({label:"radix-sort-tmp-keys",size:this.count*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST}),this.hasValues&&(this.buffers.tmpValues=this.device.createBuffer({label:"radix-sort-tmp-values",size:this.count*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST})),this.buffers.localPrefixSum=this.device.createBuffer({label:"radix-sort-local-prefix-sum",size:this.count*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST})}getPassInData(t){return{keys:t?this.buffers.keys:this.buffers.tmpKeys,values:t?this.buffers.values:this.buffers.tmpValues}}getPassOutData(t){return{keys:t?this.buffers.tmpKeys:this.buffers.keys,values:t?this.buffers.tmpValues:this.buffers.values}}createBlockSumPipeline(t,e){const r=this.device.createBindGroupLayout({label:"radix-sort-block-sum",entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:this.localShuffle?"storage":"read-only-storage"}},{binding:1,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},{binding:2,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},...this.localShuffle&&this.hasValues?[{binding:3,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}}]:[]]}),i=this.device.createBindGroup({layout:r,entries:[{binding:0,resource:{buffer:t.keys}},{binding:1,resource:{buffer:this.buffers.localPrefixSum}},{binding:2,resource:{buffer:this.buffers.prefixBlockSum}},...this.localShuffle&&this.hasValues?[{binding:3,resource:{buffer:t.values}}]:[]]}),s=this.device.createPipelineLayout({bindGroupLayouts:[r]});return{pipeline:this.device.createComputePipeline({label:"radix-sort-block-sum",layout:s,compute:{module:this.shaderModules.blockSum,entryPoint:"radix_sort",constants:{WORKGROUP_SIZE_X:this.workgroupSize.x,WORKGROUP_SIZE_Y:this.workgroupSize.y,WORKGROUP_COUNT:this.workgroupCount,THREADS_PER_WORKGROUP:this.threadsPerWorkgroup,ELEMENT_COUNT:this.count,CURRENT_BIT:e}}}),bindGroup:i}}createCheckSortKernels(t){if(!this.checkOrder)return;const{checkSortFastCount:e,checkSortFullCount:r,startFull:i}=t;this.kernels.checkSortFull=new G({mode:"full",device:this.device,data:this.data,result:this.buffers.dispatchSize,original:this.buffers.originalDispatchSize,isSorted:this.buffers.isSorted,count:r,start:i,workgroupSize:this.workgroupSize}),this.kernels.checkSortFast=new G({mode:"fast",device:this.device,data:this.data,result:this.buffers.checkSortFullDispatchSize,original:this.buffers.originalCheckSortFullDispatchSize,isSorted:this.buffers.isSorted,count:e,workgroupSize:this.workgroupSize});const s=this.initialDispatch.length/3;if(this.kernels.checkSortFast.threadsPerWorkgroup<this.kernels.checkSortFull.pipelines.length||this.kernels.checkSortFull.threadsPerWorkgroup<s){console.warn("Warning: workgroup size is too small to enable check sort optimization, disabling..."),this.checkOrder=!1;return}this.kernels.checkSortReset=new G({mode:"reset",device:this.device,data:this.data,original:this.buffers.originalDispatchSize,result:this.buffers.dispatchSize,isSorted:this.buffers.isSorted,count:s,workgroupSize:I(this.device,s)})}createReorderPipeline(t,e,r){const i=this.device.createBindGroupLayout({label:"radix-sort-reorder",entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},{binding:1,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},{binding:2,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},{binding:3,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},...this.hasValues?[{binding:4,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},{binding:5,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}}]:[]]}),s=this.device.createBindGroup({layout:i,entries:[{binding:0,resource:{buffer:t.keys}},{binding:1,resource:{buffer:e.keys}},{binding:2,resource:{buffer:this.buffers.localPrefixSum}},{binding:3,resource:{buffer:this.buffers.prefixBlockSum}},...this.hasValues?[{binding:4,resource:{buffer:t.values}},{binding:5,resource:{buffer:e.values}}]:[]]}),a=this.device.createPipelineLayout({bindGroupLayouts:[i]});return{pipeline:this.device.createComputePipeline({label:"radix-sort-reorder",layout:a,compute:{module:this.shaderModules.reorder,entryPoint:"radix_sort_reorder",constants:{WORKGROUP_SIZE_X:this.workgroupSize.x,WORKGROUP_SIZE_Y:this.workgroupSize.y,WORKGROUP_COUNT:this.workgroupCount,THREADS_PER_WORKGROUP:this.threadsPerWorkgroup,ELEMENT_COUNT:this.count,CURRENT_BIT:r}}}),bindGroup:s}}}function K(u,t,e=0){const r=u.createBuffer({size:t.length*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|e,mappedAtCreation:!0});new Uint32Array(r.getMappedRange()).set(t),r.unmap();const i=u.createBuffer({size:t.length*4,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST});return[r,i]}function ae(u){const e=u.createQuerySet({type:"timestamp",count:2}),r=u.createBuffer({size:8*2,usage:GPUBufferUsage.QUERY_RESOLVE|GPUBufferUsage.COPY_SRC}),i=u.createBuffer({size:8*2,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST});return{descriptor:{timestampWrites:{querySet:e,beginningOfPassWriteIndex:0,endOfPassWriteIndex:1}},resolve:o=>{o.resolveQuerySet(e,0,2,r,0),o.copyBufferToBuffer(r,0,i,0,8*2)},getTimestamps:async()=>{await i.mapAsync(GPUMapMode.READ);const o=new BigUint64Array(i.getMappedRange().slice());return i.unmap(),o}}}const n={elementCount:2**20,bitCount:32,workgroupSize:16,checkOrder:!1,localShuffle:!1,avoidBankConflicts:!1,sortMode:"Keys",initialSort:"Random",consecutiveSorts:1};window.onload=async function(){var i;const t=await((i=navigator.gpu)==null?void 0:i.requestAdapter()),e=await(t==null?void 0:t.requestDevice({requiredFeatures:["timestamp-query"],requiredLimits:{maxComputeInvocationsPerWorkgroup:32*32}}));if(!e)throw document.getElementById("info").innerHTML="WebGPU doesn't appear to be supported on this device.",document.getElementById("info").style.color="#ff7a48",new Error("Could not create WebGPU device");const r=new ce(e);r.onClickSort=()=>le(e)};async function ne(u,t=!0){const e=new Uint32Array(n.elementCount),r=2**n.bitCount;switch(n.initialSort){case"Random":e.forEach((g,_)=>e[_]=Math.floor(Math.random()*r));break;case"Sorted":e.forEach((g,_)=>e[_]=_);break}const[i]=K(u,e);let s;if(n.sortMode==="Keys & Values"){const g=new Uint32Array(n.elementCount).map(()=>Math.floor(Math.random()*1e6));s=K(u,g)[0]}const a=new oe({device:u,data:{keys:i,values:s},count:n.elementCount,bitCount:n.bitCount,workgroupSize:{x:n.workgroupSize,y:n.workgroupSize},checkOrder:n.checkOrder,localShuffle:n.localShuffle,avoidBankConflicts:n.avoidBankConflicts}),o=ae(u),l=u.createCommandEncoder(),f=l.beginComputePass(o.descriptor);a.dispatch(f),f.end(),o.resolve(l),u.queue.submit([l.finish()]);const p=await o.getTimestamps(),m={cpu:0,gpu:Number(p[1]-p[0])/1e6};if(t){const g=performance.now();e.sort((_,R)=>_-R),m.cpu=performance.now()-g}return m}async function le(u){let t=0,e=0;const r=document.getElementById("results");r.children[0].id==="info"&&(r.innerHTML="");const i=c(r,"div","result");i.innerHTML+=`[${r.children.length}] `,i.innerHTML+=`Sorting ${S(y(n.elementCount),"#ff9933")} ${n.sortMode.toLowerCase()} of ${S(n.bitCount.toString(),"#ff9933")} bits`,i.innerHTML+=`<br>Initial sort: ${n.initialSort}, Workgroup size: ${n.workgroupSize}x${n.workgroupSize}`,i.innerHTML+=`<br>Optimizations: (${n.checkOrder}, ${n.localShuffle}, ${n.avoidBankConflicts})`;for(let a=0;a<n.consecutiveSorts;a+=1){const o=await ne(u,a==0);t+=o.cpu,e+=o.gpu}const s=e/n.consecutiveSorts;i.innerHTML+=`<br>> CPU Reference: ${S(t.toFixed(2)+"ms","#abff33")}, `,i.innerHTML+=`GPU Average (${S(n.consecutiveSorts.toString(),"#ff9933")} sorts): ${S(s.toFixed(2)+"ms","#abff33")}, `,i.innerHTML+=`Speedup: ${S("x"+(t/s).toFixed(2),t/s>=1?"#abff33":"#ff3333")}<br><br>`,i.scrollIntoView({behavior:"smooth",block:"end"})}class ce{constructor(t){d(this,"dom");d(this,"onClickSort",()=>{});this.dom=document.getElementById("gui"),this.createHeader(),this.createTitle("Radix Sort Kernel"),this.createSlider(n,"elementCount","Element Count",n.elementCount,1e4,2**24,1,{logarithmic:!0}),this.createSlider(n,"bitCount","Bit Count",n.bitCount,4,32,4),this.createSlider(n,"workgroupSize","Workgroup Size",n.workgroupSize,2,5,1,{power_of_two:!0}),this.createTitle("Optimizations"),this.createCheckbox(n,"checkOrder","Check If Sorted",n.checkOrder),this.createCheckbox(n,"localShuffle","Local Shuffle",n.localShuffle),this.createCheckbox(n,"avoidBankConflicts","Avoid Bank Conflicts",n.avoidBankConflicts),this.createTitle("Testing"),this.createDropdown(n,"initialSort","Initial Sort",["Random","Sorted"]),this.createDropdown(n,"sortMode","Sort Mode",["Keys","Keys & Values"]),this.createSlider(n,"consecutiveSorts","Consecutive Sorts",n.consecutiveSorts,1,20,1),this.createButton("Run Radix Sort",()=>this.onClickSort(t)),this.addHints()}addHints(){const t=this.dom.querySelectorAll(".gui-ctn"),e=["","Number of elements to sort","Number of bits to sort","Workgroup size in x and y dimensions","","Check if the data is sorted after each pass to stop the kernel early","Use local shuffle optimization (does not seem to improve performance)","Avoid bank conflicts in shared memory (does not seem to improve performance)","","Initial order of the elements","Whether to use values in addition to keys","Number of consecutive sorts to run","Run the Radix Sort kernel !"];t.forEach((r,i)=>{r.title=e[i]})}createTitle(t){const e=c(this.dom,"div","gui-ctn");c(e,"div","gui-title",{innerText:t})}createHeader(){const t=c(this.dom,"div","gui-header"),e=c(t,"div","gui-icon"),r=c(t,"a","gui-link",{href:"https://github.com/kishimisu/WebGPU-Radix-Sort",innerText:"https://github.com/kishimisu/WebGPU-Radix-Sort"}),i='<svg role="img" viewBox="0 0 24 24" fill="white" xmlns="http://www.w3.org/2000/svg"><title>GitHub</title><path d="M12 .297c-6.63 0-12 5.373-12 12 0 5.303 3.438 9.8 8.205 11.385.6.113.82-.258.82-.577 0-.285-.01-1.04-.015-2.04-3.338.724-4.042-1.61-4.042-1.61C4.422 18.07 3.633 17.7 3.633 17.7c-1.087-.744.084-.729.084-.729 1.205.084 1.838 1.236 1.838 1.236 1.07 1.835 2.809 1.305 3.495.998.108-.776.417-1.305.76-1.605-2.665-.3-5.466-1.332-5.466-5.93 0-1.31.465-2.38 1.235-3.22-.135-.303-.54-1.523.105-3.176 0 0 1.005-.322 3.3 1.23.96-.267 1.98-.399 3-.405 1.02.006 2.04.138 3 .405 2.28-1.552 3.285-1.23 3.285-1.23.645 1.653.24 2.873.12 3.176.765.84 1.23 1.91 1.23 3.22 0 4.61-2.805 5.625-5.475 5.92.42.36.81 1.096.81 2.22 0 1.606-.015 2.896-.015 3.286 0 .315.21.69.825.57C20.565 22.092 24 17.592 24 12.297c0-6.627-5.373-12-12-12"/></svg>';e.innerHTML=i,r.target="_blank"}createSlider(t,e,r,i=0,s=0,a=1,o=.01,{logarithmic:l=!1,power_of_two:f=!1}={}){const p=Math.log(s),g=(Math.log(a)-p)/(a-s),_=h=>l?Math.round(Math.exp(p+g*(h-s))):f?Math.pow(2,h):h,R=h=>l?(Math.log(h)-p)/g+s:f?Math.log2(h):h,E=c(this.dom,"div","gui-ctn");c(E,"label","gui-label",{innerText:r});const W=c(E,"div","gui-input"),D=c(W,"input","gui-slider",{type:"range",min:s,max:a,step:o,value:R(i)}),Y=c(W,"span","gui-value",{innerText:y(i)});D.addEventListener("input",()=>{let h=parseFloat(D.value);(l||f)&&(h=_(parseFloat(D.value))),t[e]=h,Y.innerText=y(h)})}createCheckbox(t,e,r,i=!1){const s=c(this.dom,"div","gui-ctn");c(s,"label","gui-label",{innerText:r});const a=c(s,"div","gui-input"),o=c(a,"input","gui-checkbox",{type:"checkbox",checked:i}),l=c(a,"span","gui-value",{innerText:i?"true":"false"});o.addEventListener("change",()=>{t[e]=o.checked,l.innerText=o.checked?"true":"false"})}createDropdown(t,e,r,i){const s=c(this.dom,"div","gui-ctn");c(s,"label","gui-label",{innerText:r});const a=c(s,"div","gui-input"),o=c(a,"select","gui-select"),l=c(a,"span","gui-value",{innerText:t[e]});i.forEach((f,p)=>{const m=c(o,"option","",{innerText:f,value:p});f===t[e]&&(m.selected=!0)}),o.addEventListener("change",()=>{t[e]=i[parseInt(o.value)],l.innerText=i[parseInt(o.value)]})}createButton(t,e){const r=c(this.dom,"div","gui-ctn");c(r,"button","gui-button",{innerText:t}).addEventListener("click",e)}}function c(u,t,e,r={}){const i=document.createElement(t);return i.className=e,Object.keys(r).forEach(s=>{i[s]=r[s]}),u.appendChild(i),i}const S=(u,t)=>`<span style="color:${t}">${u}</span>`,y=u=>u.toString().replace(/\B(?=(\d{3})+(?!\d))/g,",");
