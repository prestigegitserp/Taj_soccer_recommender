import test from "node:test";
import assert from "node:assert/strict";
import {FEATURE_NAMES,featuresFor,prepareInput,temperatureAdjust} from "../docs/football-features.mjs";

test("A true 20-feature vector is computed only from prior finished matches",()=>{
  const all=[];
  for(let i=0;i<40;i++){
    all.push({id:String(i),league:"eng.1",kickoff:new Date(Date.UTC(2025,0,1+i*3)).toISOString(),
       state:"post",home:{id:String(i%4+1),score:i%4},
       away:{id:String((i+2)%4+1),score:(3*i+1)%3}});
  }
  const match={id:"future",league:"eng.1",kickoff:"2025-06-15T19:00:00Z",
    home:{id:"1"},away:{id:"3"}};
  const first=featuresFor(match,all);
  assert.equal(first?.length,20);
  all.push({id:"leak",league:"eng.1",kickoff:"2025-06-16T19:00:00Z",
     state:"post",home:{id:"1",score:15},away:{id:"3",score:0}});
  assert.deepEqual(featuresFor(match,all),first);
});
test("No invented AI inputs from five upcoming fixtures with no historical scores",()=>{
  const game={id:"2026",league:"eng.1",kickoff:"2026-10-21T15:00:00Z",home:{id:"11"},away:{id:"22"}};
  assert.equal(featuresFor(game,[]),null);
});
test("feature schema version and normalization must match exact names",()=>{
  const card={schema:"taj-litert-1x2-v1",input_features:FEATURE_NAMES,
    mean:Array(20).fill(1),std:Array(20).fill(2)};
  const arr=prepareInput(Array(20).fill(3),card);
  assert.ok(arr instanceof Float32Array);
  assert.equal(arr[0],1);
  assert.equal(prepareInput(Array(19).fill(0),card),null);
  assert.equal(prepareInput(Array(20).fill(0),{...card,input_features:["bad",...FEATURE_NAMES.slice(1)]}),null);
});
test("temperature-adjusted probabilities remain in simplex",()=>{
  for(const t of [.7,1,1.25,2]){
    const p=temperatureAdjust([.55,.23,.22],t);
    assert.ok(Math.abs(p.reduce((a,b)=>a+b,0)-1)<1e-12);
    assert.ok(p.every(x=>x>0&&x<1));
  }
  assert.equal(temperatureAdjust([.8,NaN,.2],1),null);
});
