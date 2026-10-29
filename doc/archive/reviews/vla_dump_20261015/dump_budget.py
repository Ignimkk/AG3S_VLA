"""Estimate uncompressed array payload, not collector runtime or compressed file size."""
import argparse,json,math
from pathlib import Path


def main():
    ap=argparse.ArgumentParser();ap.add_argument('inventory',type=Path);ap.add_argument('output',type=Path);args=ap.parse_args()
    entries=json.loads(args.inventory.read_text())
    source={}
    for e in entries:source[(e['file'],e['key'])]=e
    def file_bytes(name):return round(sum(e['mb']*1e6 for e in entries if e['file']==name))
    parts={name:file_bytes(name) for name in ['inputs.npz','embed.npz','outputs.npz','derived.npz']}
    shapes={
        'siglip_encoded':[3,256,1152],
        'prefix_resid_out_L4_L8_L12':[3,968,2048],
        'prefix_final_hidden':[968,2048],
        'prefix_KV_cache_canonical':[2,18,968,256],
        'prefix_attention_named_rows_3':[18,8,3,968],
        'suffix_attention_step0_all':[18,8,50,1018],
        'suffix_attention_L8H2_steps1_to9':[9,50,1018],
        'suffix_V_all_steps':[10,18,50,256],
    }
    for k,shape in shapes.items():parts[k]=math.prod(shape)*2
    parts['flow_xt_vt_time']=10*(2*50*32*4+4)
    core=sum(parts.values())
    survey=core-parts['suffix_attention_L8H2_steps1_to9']+9*18*8*50*1018*2
    full=round(sum(e['mb']*1e6 for e in entries))
    geometry={'registered_depth_float32_3x480x640':3*480*640*4,'raw_rgb_uint8_3x480x640':3*480*640*3}
    result={
        'status':'design_budget_not_implemented_collector',
        'source_inventory':str(args.inventory),'units':'bytes and decimal MB; uncompressed tensor payload',
        'assumptions':{'cameras':3,'patches_per_camera':256,'prefix_slots':968,'named_query_rows':3,'layers':18,'query_heads':8,'kv_heads':1,'action_horizon':50,'action_dim':32,'denoise_steps':10,'bulk_dtype':'float16; retain original dtype separately if float16 overflows'},
        'core_components_bytes':parts,'shape_components_fp16':shapes,
        'profiles':{name:{'bytes':size,'decimal_MB':size/1e6,'decimal_GB_for_100_frames':size*100/1e9,'fraction_of_full':size/full} for name,size in [('core',core),('survey',survey),('full_current',full)]},
        'optional_geometry_payload_bytes':geometry,
        'excluded':'container/header/metadata overhead, GT, extra raw observations, compression ratio, normals/point clouds, variable noun spans',
        'note':'Store geometry/raw observations once by immutable reference when already recorded; do not duplicate generated point clouds.'
    }
    args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result['profiles'],indent=2))

if __name__=='__main__':main()
