#!/usr/bin/env python3
import io
import json
import pathlib
import struct
import sys
import numpy as np
from PIL import Image
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
import glb_metrics as glb


def canonical(points):
    a=np.ascontiguousarray(points,dtype=np.float32)
    order=np.argsort(a.view(np.dtype([('x','<f4'),('y','<f4'),('z','<f4')])).reshape(-1,3),axis=1,order=['x','y','z'])
    keys=np.ascontiguousarray(np.take_along_axis(a,order[:,:,None],axis=1)).reshape(-1,9).view('V36').reshape(-1)
    parity=sum(order[:,i]>order[:,j] for i,j in [(0,1),(0,2),(1,2)])%2
    return keys,order,parity


asset,imported=map(pathlib.Path,sys.argv[1:3])
g,b=glb.parse_glb(asset)
assert len(g['meshes'])==1 and len(g['meshes'][0]['primitives'])==1
primitive=g['meshes'][0]['primitives'][0];attr=primitive['attributes']
gp=glb.read_accessor(g,b,attr['POSITION']);gu=glb.read_accessor(g,b,attr['TEXCOORD_0']);gn=glb.read_accessor(g,b,attr['NORMAL']);gf=glb.read_accessor(g,b,primitive['indices']).reshape(-1,3)
blob=imported.read_bytes();nv,nf=struct.unpack_from('<II',blob);assert len(blob)==8+nv*20+nf*12
p=np.frombuffer(blob,'<f4',nv*3,8).reshape(-1,3);uv=np.frombuffer(blob,'<f4',nv*2,8+nv*12).reshape(-1,2);f=np.frombuffer(blob,'<i4',nf*3,8+nv*20).reshape(-1,3)
n=np.frombuffer(pathlib.Path(str(imported)+'.normals.bin').read_bytes(),'<f4').reshape(-1,3);assert len(n)==nv
gk,go,gpar=canonical(gp[gf]);ok,oo,opar=canonical(p[f]);order=np.argsort(gk);sorted_keys=gk[order]
assert len(gk)==len(ok) and np.all(sorted_keys[1:]!=sorted_keys[:-1])
match=np.searchsorted(sorted_keys,ok);assert np.all(match<len(gk));match=order[match]
assert np.array_equal(gk[match],ok),'OBJ geometry changed'
assert np.array_equal(gpar[match],opar),'OBJ winding changed'
for name,ga,oa in [('uv',gu,uv),('normal',gn,n)]:
    expected=np.take_along_axis(ga[gf],go[:,:,None],axis=1)[match]
    actual=np.take_along_axis(oa[f],oo[:,:,None],axis=1)
    assert np.array_equal(expected,actual),(name,float(np.max(np.abs(expected-actual))))
images=[np.array(Image.open(io.BytesIO(glb.image_bytes(g,b,img))).convert('RGBA')) for img in g['images']]
stem=asset.with_suffix('').name
if not (asset.parent/(stem+'_base.png')).exists():
    stem += '_quads'
channels=[('base',images[0]),('roughness',images[1][:,:,1]),('metallic',images[1][:,:,2])]
if len(images)>2:
    channels.append(('normal',images[2]))
for suffix,expected in channels:
    actual=np.array(Image.open(asset.parent/(stem+'_'+suffix+'.png')))
    assert np.array_equal(actual[::-1],expected),suffix+' image origin/channel mismatch'
result={'triangles':nf,'geometry_exact':True,'winding_exact':True,'uv_float32_exact':True,'normals_float32_exact':True,'all_texture_pixels_exact_after_image_origin_conversion':True,'quad_export_roundtrip_pass':True,'complete_pipeline_accepted':False}
asset.with_suffix('.quad-export.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
