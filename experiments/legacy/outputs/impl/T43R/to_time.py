"""T43 R: TO 시간 — 같은 기록 청크를 bare / cover 모델로 풀어 벽시계와 단계별 시간을 잰다 (반복 수 = 기록)."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, logging, sys, time, pathlib, numpy as np, mujoco
logging.disable(logging.WARNING)
import AG3S.trajopt.experiments.fixed_input_ablation as fia
from AG3S.ag3s.experiments.reports.grounding_report import build_constraint_robot_model, GRIPPER_LINKS
from AG3S.ag3s.experiments.sources.mujoco_source import TransportScene
from AG3S.trajopt.types import ChunkLayout
from AG3S.trajopt.refiner import TrajOptChunkRefiner
cover_opt = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
S=_ag3s_workspace_path('outputs/verify/T40/G/runs/E3b/server_8240_20261003T135313')
RUNS={"C1807":(S+'/server_constraints/run_0000', _ag3s_workspace_path('outputs/verify/T40/G/runs/E3b/C/E3b_ep1807_s18071'), range(119,131)),
      "B1968":(S+'/server_constraints/run_0000', _ag3s_workspace_path('outputs/verify/T40/G/runs/E3b/B/E3b_ep1968_s19681'), range(140,152))}
M=_ag3s_workspace_path('robots/rby1/description/models/rby1a/mujoco/model_transport.xml')
m=mujoco.MjModel.from_xml_path(M); scene=TransportScene.attach(m, mujoco.MjData(m))
opts=dict(sphere_spacing=0.4,max_spheres_per_capsule=32,capsule_radius_scale=0.05)
models={"bare":build_constraint_robot_model(scene,link_filter=GRIPPER_LINKS,sphere_options=opts)}
for name,o in cover_opt.items():
    models[name]=build_constraint_robot_model(scene,link_filter=GRIPPER_LINKS,sphere_options=opts,gripper_cover=o or True)
qadr={mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_JOINT,i): int(m.jnt_qposadr[i]) for i in range(m.njnt)}
out={}
for run,(cdir,rdir,chunks) in RUNS.items():
    args=fia.build_parser().parse_args(["--constraints",cdir,"--frames",rdir+"/frames.jsonl","--out","/dev/null","--server-log",S+"/serve_safe.log"])
    manifest=json.loads(pathlib.Path(rdir+"/manifest.json").read_text())
    log=fia.parse_server_log(S+"/serve_safe.log"); st=fia.resolve_settings(args,log,manifest)
    rows=[json.loads(l) for l in open(rdir+"/frames.jsonl") if l.strip()]
    plan={int(r["seq"]):r for r in rows if r.get("kind")=="planning"}
    for mname,robot in models.items():
        layout=ChunkLayout.rby1(robot.joint_names)
        for k in chunks:
            z=np.load(f"{cdir}/chunk_{k:05d}.npz"); s=json.loads(str(z["summary_json"]))
            row=plan.get(int(s["t_step"]))
            if row is None or np.abs(np.asarray(row["actions"])-z["refined_chunk"]).max()>1e-5: continue
            q_now=fia.q_now_from_qpos(row["qpos"],qadr)
            field=fia.build_field(z,outside_distance=0.4)
            tp=z["target_points"] if "target_points" in z.files else None
            margin,mask,auth,_,_=fia.contact_policy(st,robot.sphere_link_names,has_target=bool(s.get("has_target")) and tp is not None,has_free=bool(field.target_free_layers),attached=False)
            from AG3S.trajopt.linearize import CollisionLinearizer
            radii=np.asarray(robot.radii)
            sc=fia.make_scene(field,radii,st.esdf_margin,target_points=tp,margin=margin,mask=mask)
            iters=int((s.get("to") or {}).get("sqp_iterations") or 1)
            cfg=fia.to_config(st,collision=st.collision_enabled,cost=st.cost,sqp={"time_budget_ms":fia.NO_CLOCK_MS,"max_iterations":iters})
            ref=TrajOptChunkRefiner(robot,layout,lambda _c: (sc,q_now,True),cfg)
            # finger joints from the record
            fj=(s.get("finger_joints") or {}).get("joints")
            if fj: robot.set_joint_parameters(fj)
            best=None
            for _rep in range(3):
                ref.reset(); t=time.perf_counter(); ref.refine(np.asarray(z["reference_chunk"],np.float64),{"t_step":int(s["t_step"]),"previous_physical_chunk":None}); w_=(time.perf_counter()-t)*1e3
                r_=ref.last_result; tm_=r_.metrics.get("timing_ms"); tm_=eval(tm_) if isinstance(tm_,str) else tm_
                if best is None or w_<best[0]: best=(w_,r_,tm_)
            wall,res,tm=best; met=res.metrics
            lin=ref.optimizer.linearizer
            traj=res.trajectory
            cl=lin.esdf_clearance(traj,q_now,sc)
            names=np.array(robot.sphere_link_names); left=np.isin(names,["ee_left","ee_finger_l1","ee_finger_l2"])
            out.setdefault(run,{}).setdefault(mname,[]).append(dict(k=k,t=int(s["t_step"]),wall_ms=wall,timing=tm,iters=met.get("sqp_iterations"),
                 final_violation_mm=float(met.get("final_violation_m",met.get("max_violation_m",0)) or 0)*1e3 if isinstance(met.get("final_violation_m",met.get("max_violation_m",0)),(int,float)) else None,
                 refined_min_left_mm=float(cl[:, left].min()*1e3), refined_min_left_base_mm=float(cl[:, :models['bare'].n_spheres][:, left[:models['bare'].n_spheres]].min()*1e3),
                 status=str(res.status) if hasattr(res,"status") else None))
            robot.set_joint_parameters(None)
            z.close()
        r=out[run][mname]
        w=np.array([x["wall_ms"] for x in r]); 
        comp={c: float(np.median([x["timing"][c] for x in r if x["timing"]])) for c in ("linearize","assemble","qp","check")}
        print(f"{run} {mname} n_spheres {robot.n_spheres} chunks {len(r)} wall median {np.median(w):.1f} p90 {np.percentile(w,90):.1f} ms  components(median) {comp}", flush=True)
json.dump(out,open(_ag3s_workspace_path('outputs/impl/T43R/to_time.json'),"w"),indent=1,default=str)
