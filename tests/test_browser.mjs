import test from "node:test";
import assert from "node:assert/strict";
import {flatten,nextFive,analyze,advice,scoreProbabilities,normalizeEspn} from "../docs/app.mjs";

const game=(id,date,state,h="H",a="A",hs=null,as=null)=>({
 id,league:"eng.1",kickoff:date,state,
 home:{id:h,name:h,short:h,score:hs},
 away:{id:a,name:a,short:a,score:as}
});
test("five upcoming are real, sorted and filtered by league",()=>{
 const games=[game("8","2026-10-25T12:00:00Z","pre"),
  game("2","2026-10-12T18:00:00Z","pre"),
  game("3","2026-10-13T18:00:00Z","pre"),
  game("4","2026-10-14T18:00:00Z","pre"),
  game("5","2026-10-15T18:00:00Z","pre"),
  game("6","2026-10-16T18:00:00Z","pre"),
  game("past","2026-10-01T18:00:00Z","post")];
 const selected=nextFive(games,"all",Date.parse("2026-10-10"));
 assert.equal(selected.length,5);
 assert.deepEqual(selected.map(x=>x.id),["2","3","4","5","6"]);
});
test("no fixtures are fabricated from empty source",()=>{
 assert.deepEqual(flatten({leagues:{}}),[]);
 assert.deepEqual(nextFive([]),[]);
});
test("browser estimates abstain if historical sample inadequate",()=>{
 const match=game("next","2026-10-20T20:00:00Z","pre");
 const a=analyze(match,[]);
 assert.equal(a.forecast,null);
 assert.equal(advice(a).length,1);
});
test("Poisson outcome components sum to one",()=>{
 const x=scoreProbabilities(1.4,.9);
 assert.ok(Math.abs(x.home+x.draw+x.away-1)<1e-9);
 assert.ok(x.home>x.away);
});
test("future scores cannot leak into pre-game analysis",()=>{
 const target=game("next","2026-10-20T20:00:00Z","pre","H","A");
 const historical=[];
 for(let i=0;i<12;i++){
  historical.push(game(String(i),"2026-10-"+String(i+2).padStart(2,"0")+"T15:00:00Z","post",
   i%2?"H":"A",i%2?"A":"H",i%3,1));
 }
 const baseline=analyze(target,historical);
 assert.ok(baseline.forecast);
 historical.push(game("future","2026-10-21T15:00:00Z","post","H","A",20,0));
 assert.deepEqual(analyze(target,historical).forecast,baseline.forecast);
});
test("ESPN home/away order must use roles, not array order",()=>{
 const x={id:"abc",date:"2026-10-21T12:00:00Z",competitions:[{competitors:[
   {homeAway:"away",team:{id:2,name:"Away"}},
   {homeAway:"home",team:{id:1,name:"Home"}}
 ]}]};
 const r=normalizeEspn(x,"eng.1");
 assert.equal(r.home.name,"Home");
 assert.equal(r.away.id,"2");
});
