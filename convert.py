#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gokoba 3D-Viewer — Konvertierungs-Pipeline
===========================================
Wandelt eine Advance-Steel-STEP-Datei in einen eigenständigen Web-3D-Viewer (HTML) um.

Ablauf:
  STEP  →  OpenCascade (Meshing + Farben pro Bauteil, sRGB)
        →  crease-angle Normalen (40°, scharfe Kanten / glatte Rundungen)
        →  Farb-Boost (HSL S*1.12, dann L*0.78 gesamt — Gokoba-Look)
        →  Z-up → Y-up, mm → m, zentrieren  →  GLB
        →  gltfpack (Meshopt -vp 16 -vn 16)  →  komprimiertes GLB
        →  HTML mit eingebettetem Viewer (Three.js, Farbe/Grau-Toggle,
           Schatten an, 20-Tage-Ablauf)

Aufruf:
  python convert.py --input model.stp --output index.html \
                    --assets-dir assets --model-name "Modellbereich" \
                    --expiry-days 20 [--gltfpack gltfpack]
"""
import argparse, base64, datetime, os, subprocess, sys, tempfile
import numpy as np


# ════════════════════════════════════════════════════════════════════
#  1) OpenCascade: Meshing + Farbe pro Solid
# ════════════════════════════════════════════════════════════════════
def mesh_with_colors(step_path, deflection=0.15, angular=0.1):
    from OCP.STEPCAFControl import STEPCAFControl_Reader
    from OCP.XCAFDoc import XCAFDoc_DocumentTool, XCAFDoc_ColorType
    from OCP.TDocStd import TDocStd_Document
    from OCP.TCollection import TCollection_ExtendedString
    from OCP.TDF import TDF_LabelSequence
    from OCP.Quantity import Quantity_Color
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopAbs import TopAbs_SOLID, TopAbs_FACE, TopAbs_REVERSED
    from OCP.TopoDS import TopoDS
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.TopLoc import TopLoc_Location
    from OCP.BRep import BRep_Tool

    doc = TDocStd_Document(TCollection_ExtendedString("doc"))
    reader = STEPCAFControl_Reader()
    reader.SetColorMode(True)
    reader.ReadFile(step_path)
    reader.Transfer(doc)
    shape_tool = XCAFDoc_DocumentTool.ShapeTool_s(doc.Main())
    color_tool = XCAFDoc_DocumentTool.ColorTool_s(doc.Main())
    labels = TDF_LabelSequence()
    shape_tool.GetFreeShapes(labels)

    def lin2srgb(c):
        return [round(x ** (1 / 2.2) * 255) for x in c]

    ctypes = [XCAFDoc_ColorType.XCAFDoc_ColorSurf, XCAFDoc_ColorType.XCAFDoc_ColorGen]
    verts, faces, colors = [], [], []
    offset = 0
    n_solids = 0

    for li in range(1, labels.Length() + 1):
        top = shape_tool.GetShape_s(labels.Value(li))
        exp = TopExp_Explorer(top, TopAbs_SOLID)
        while exp.More():
            solid = exp.Current()
            n_solids += 1
            col = Quantity_Color()
            found = False
            for ct in ctypes:
                if color_tool.GetColor(solid, ct, col):
                    found = True
                    break
            rgb = lin2srgb([col.Red(), col.Green(), col.Blue()]) if found else [128, 128, 128]
            rgba = rgb + [255]

            BRepMesh_IncrementalMesh(solid, deflection, False, angular, True)
            fexp = TopExp_Explorer(solid, TopAbs_FACE)
            while fexp.More():
                face = TopoDS.Face_s(fexp.Current())
                loc = TopLoc_Location()
                tri = BRep_Tool.Triangulation_s(face, loc)
                if tri is not None:
                    trsf = loc.Transformation()
                    rev = (face.Orientation() == TopAbs_REVERSED)
                    nb = tri.NbNodes()
                    for i in range(1, nb + 1):
                        p = tri.Node(i).Transformed(trsf)
                        verts.append([p.X(), p.Y(), p.Z()])
                    for i in range(1, tri.NbTriangles() + 1):
                        t = tri.Triangle(i)
                        a, b, c = t.Get()
                        if rev:
                            b, c = c, b
                        faces.append([offset + a - 1, offset + b - 1, offset + c - 1])
                    colors.extend([rgba] * nb)
                    offset += nb
                fexp.Next()
            exp.Next()

    print(f"  Solids: {n_solids}, Vertices: {len(verts):,}, Faces: {len(faces):,}")
    return (np.array(verts, dtype=np.float64),
            np.array(faces, dtype=np.int64),
            np.array(colors, dtype=np.uint8))


# ════════════════════════════════════════════════════════════════════
#  2) crease-angle Normalen + Farb-Boost + Orientierung → GLB
# ════════════════════════════════════════════════════════════════════
def build_glb(verts, faces, colors, out_glb, crease_deg=40):
    import trimesh
    import colorsys
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components

    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    face_colors = colors[faces[:, 0]]

    # --- crease-angle Normalen ---
    CREASE = np.radians(crease_deg)
    fn = mesh.face_normals
    nf = len(faces)
    adj = mesh.face_adjacency
    ang = mesh.face_adjacency_angles
    smooth = adj[ang < CREASE]
    if len(smooth) > 0:
        data = np.ones(len(smooth))
        g = csr_matrix((data, (smooth[:, 0], smooth[:, 1])), shape=(nf, nf))
        ng, fg = connected_components(g, directed=False)
    else:
        ng, fg = nf, np.arange(nf)

    corners = faces.reshape(-1)
    cf = np.repeat(np.arange(nf), 3)
    cg = fg[cf]
    keys = corners.astype(np.int64) * np.int64(ng) + cg.astype(np.int64)
    uk, inv = np.unique(keys, return_inverse=True)
    nn = len(uk)
    pos = np.zeros((nn, 3)); pos[inv] = verts[corners]
    areas = mesh.area_faces; w = areas[cf][:, None]
    nrm = np.zeros((nn, 3)); np.add.at(nrm, inv, fn[cf] * w)
    L = np.linalg.norm(nrm, axis=1, keepdims=True); L[L == 0] = 1; nrm /= L
    col = np.zeros((nn, 4), dtype=np.uint8); col[inv] = face_colors[cf]
    nfaces = inv.reshape(-1, 3)

    # --- Farb-Boost (Gokoba-Look: S*1.12, L*0.78 gesamt) ---
    def boost(rgb):
        r, g, b = [x / 255 for x in rgb[:3]]
        h, l, s = colorsys.rgb_to_hls(r, g, b)
        s = min(1.0, s * 1.12)
        l = l * 0.78
        r2, g2, b2 = colorsys.hls_to_rgb(h, l, s)
        return [round(r2 * 255), round(g2 * 255), round(b2 * 255)]

    uniq = np.unique(col.reshape(-1, 4), axis=0)
    for c in uniq:
        key = (c[0], c[1], c[2]); nb = boost(c)
        m = (col[:, 0] == key[0]) & (col[:, 1] == key[1]) & (col[:, 2] == key[2])
        col[m, 0] = nb[0]; col[m, 1] = nb[1]; col[m, 2] = nb[2]

    m = trimesh.Trimesh(vertices=pos, faces=nfaces, process=False)
    m.visual.vertex_colors = col
    m.vertex_normals = nrm
    # Z-up → Y-up, mm → m, zentrieren
    m.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0]))
    m.apply_scale(0.001)
    ctr = m.bounds[0] + (m.bounds[1] - m.bounds[0]) / 2
    m.apply_translation(-ctr)
    m.export(out_glb)
    print(f"  GLB (unkomprimiert): {os.path.getsize(out_glb)/1024/1024:.1f} MB, {len(uniq)} Farben")


# ════════════════════════════════════════════════════════════════════
#  3) gltfpack (Meshopt-Kompression)
# ════════════════════════════════════════════════════════════════════
def compress_glb(in_glb, out_glb, gltfpack="gltfpack"):
    cmd = [gltfpack, "-i", in_glb, "-o", out_glb, "-cc", "-vp", "16", "-vn", "16", "-vc", "8"]
    subprocess.run(cmd, check=True, capture_output=True)
    print(f"  GLB (komprimiert):  {os.path.getsize(out_glb)/1024/1024:.1f} MB")


# ════════════════════════════════════════════════════════════════════
#  4) HTML-Viewer erzeugen (Three.js eingebettet, 20-Tage-Ablauf)
# ════════════════════════════════════════════════════════════════════
def build_html(glb_path, assets_dir, model_name, expiry_iso, out_html):
    def rd(p):
        with open(os.path.join(assets_dir, p), "r", encoding="utf-8") as f:
            return f.read()

    three_js = rd("three.min.js")
    gltf_js = rd("GLTFLoader.js")
    orbit_js = rd("OrbitControls.js")
    meshopt_js = rd("meshopt_decoder.js")
    with open(glb_path, "rb") as f:
        glb_b64 = base64.b64encode(f.read()).decode()

    html = HTML_TEMPLATE
    html = html.replace("__MODEL_NAME__", model_name)
    html = html.replace("__EXPIRY_ISO__", expiry_iso)
    html = html.replace("/*__THREE__*/", three_js)
    html = html.replace("/*__GLTF__*/", gltf_js)
    html = html.replace("/*__ORBIT__*/", orbit_js)
    html = html.replace("/*__MESHOPT__*/", meshopt_js)
    html = html.replace("__GLB_B64__", glb_b64)

    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  HTML: {os.path.getsize(out_html)/1024/1024:.1f} MB")


HTML_TEMPLATE = r'''<!DOCTYPE html>
<html lang="de"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Gokoba 3D-Viewer</title>
<style>
*{box-sizing:border-box;margin:0;padding:0;}
:root{--sand:#F7F3EE;--acc:#D4622A;--ink:#1C1C1C;--ink2:#6B6560;--brd:#D9D2C8;}
body{font-family:Inter,system-ui,sans-serif;background:var(--sand);height:100vh;overflow:hidden;}
header{position:fixed;top:0;left:0;right:0;height:60px;z-index:100;background:#fff;
  border-bottom:1px solid var(--brd);display:flex;align-items:center;justify-content:center;padding:0 22px;}
.header-inner{width:100%;max-width:820px;display:flex;align-items:center;justify-content:space-between;gap:16px;}
.logo{font-size:20px;font-weight:600;letter-spacing:.16em;text-transform:uppercase;white-space:nowrap;}
.logo em{color:var(--acc);font-style:normal;}
.logo-sub{font-size:13px;font-weight:400;letter-spacing:.06em;color:var(--ink2);text-transform:none;}
.controls{display:flex;gap:10px;align-items:center;flex-wrap:wrap;justify-content:flex-end;}
.toggle{display:flex;border:1px solid var(--brd);border-radius:8px;overflow:hidden;}
.tbtn{font-family:inherit;font-size:11px;font-weight:500;padding:7px 14px;
  border:none;background:#fff;color:var(--ink2);cursor:pointer;transition:all .15s;white-space:nowrap;}
.tbtn.active{background:var(--acc);color:#fff;}
#vpanel{height:100vh;padding-top:60px;position:relative;}
#tcv{width:100%;height:100%;display:block;}
#ldg{position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;
  justify-content:center;gap:14px;background:var(--sand);z-index:10;padding:24px;text-align:center;}
#ldg.hide{display:none;}
.spin{width:44px;height:44px;border:3px solid var(--brd);border-top-color:var(--acc);
  border-radius:50%;animation:sp .8s linear infinite;}
@keyframes sp{to{transform:rotate(360deg);}}
.ldg-txt{font-size:12px;color:var(--ink2);}
#expired{position:absolute;inset:0;display:none;flex-direction:column;align-items:center;
  justify-content:center;gap:12px;background:var(--sand);z-index:20;padding:32px;text-align:center;}
#expired h2{font-size:20px;color:var(--ink);font-weight:600;}
#expired p{font-size:13px;color:var(--ink2);max-width:420px;line-height:1.6;}
#bar{position:absolute;bottom:0;left:0;right:0;background:rgba(247,243,238,.94);
  backdrop-filter:blur(8px);border-top:1px solid var(--brd);padding:10px 22px;display:flex;justify-content:center;}
.bar-inner{width:100%;max-width:900px;display:flex;align-items:center;justify-content:space-between;gap:16px;}
.bi{display:flex;flex-direction:column;gap:2px;}
.bi.center{text-align:center;}.bi.right{text-align:right;}
.bl{font-size:9px;color:var(--ink2);text-transform:uppercase;letter-spacing:.08em;}
.bv{font-size:15px;font-weight:600;color:var(--ink);}
.bv.sm{font-size:12px;font-weight:500;}
.hint{font-size:9px;color:var(--ink2);opacity:.55;line-height:1.5;}
</style>
</head>
<body>
<header>
  <div class="header-inner">
    <div class="logo">GOKO<em>BA</em> <span class="logo-sub">3D-Viewer</span></div>
    <div class="controls">
      <div class="toggle" id="colorToggle">
        <button class="tbtn active" data-mode="color">Farbe</button>
        <button class="tbtn" data-mode="gray">Grautöne</button>
      </div>
      <div class="toggle" id="shadeToggle">
        <button class="tbtn" data-shade="off">Schatten aus</button>
        <button class="tbtn active" data-shade="on">Schatten an</button>
      </div>
    </div>
  </div>
</header>
<div id="vpanel">
  <canvas id="tcv"></canvas>
  <div id="ldg"><div class="spin"></div><div class="ldg-txt" id="lt">Modell wird geladen…</div></div>
  <div id="expired">
    <h2>Dieser Link ist abgelaufen</h2>
    <p>Die Gültigkeit dieser 3D-Ansicht ist abgelaufen. Bitte fordern Sie bei Bedarf einen neuen Link an.</p>
  </div>
  <div id="bar">
    <div class="bar-inner">
      <div class="bi"><div class="bl">Projekt</div><div class="bv">__MODEL_NAME__</div></div>
      <div class="bi center"><div class="bl">Erstellt mit Autodesk Advance Steel</div><div class="bv sm">Paul Thomas</div></div>
      <div class="bi right"><div class="hint">Ziehen: drehen &middot; Scroll: zoomen<br>Rechtsklick: verschieben</div></div>
    </div>
  </div>
</div>
<script>
// ── 20-Tage-Ablaufprüfung ──
var EXPIRY = "__EXPIRY_ISO__";
if (EXPIRY && new Date() > new Date(EXPIRY)) {
  document.getElementById("ldg").style.display = "none";
  document.getElementById("expired").style.display = "flex";
  document.getElementById("colorToggle").style.display = "none";
  document.getElementById("shadeToggle").style.display = "none";
  throw new Error("expired");
}
</script>
<script>/*__THREE__*/</script>
<script>/*__GLTF__*/</script>
<script>/*__ORBIT__*/</script>
<script>/*__MESHOPT__*/</script>
<script>var GLB_B64="__GLB_B64__";</script>
<script>
var cv=document.getElementById("tcv"),vp=document.getElementById("vpanel"),lt=document.getElementById("lt");
var renderer=new THREE.WebGLRenderer({canvas:cv,antialias:true});
renderer.setPixelRatio(Math.min(devicePixelRatio,2));
renderer.setClearColor(0xEDE8E0);
renderer.outputEncoding=THREE.sRGBEncoding;
renderer.toneMapping=THREE.LinearToneMapping;
renderer.toneMappingExposure=0.98;
renderer.shadowMap.enabled=true;
renderer.shadowMap.type=THREE.PCFSoftShadowMap;
renderer.shadowMap.autoUpdate=false;
var scene=new THREE.Scene();
var cam=new THREE.PerspectiveCamera(42,1,0.01,5000);
var ctrl=new THREE.OrbitControls(cam,cv);
ctrl.enableDamping=true;ctrl.dampingFactor=0.07;
var ambient=new THREE.AmbientLight(0xffffff,0.22);scene.add(ambient);
var key=new THREE.DirectionalLight(0xffffff,0.82);scene.add(key);
var fill=new THREE.DirectionalLight(0xffffff,0.16);fill.position.set(-5,2,-4);scene.add(fill);
var meshes=[],origColors=[],grayColors=[],ground=null,useShade=true;
function buildGray(ca){var arr=ca.array,T=arr.constructor,ga=new T(arr.length),is=ca.itemSize;
  var isInt=(arr instanceof Uint8Array||arr instanceof Uint16Array||arr instanceof Int8Array||arr instanceof Int16Array);
  for(var i=0;i<arr.length;i+=is){var lum=0.299*arr[i]+0.587*arr[i+1]+0.114*arr[i+2];if(isInt)lum=Math.round(lum);
    ga[i]=lum;ga[i+1]=lum;ga[i+2]=lum;if(is>=4)ga[i+3]=arr[i+3];}
  return new THREE.BufferAttribute(ga,is,ca.normalized);}
function setColorMode(mode){meshes.forEach(function(m,i){
  m.geometry.setAttribute('color',mode==='gray'?grayColors[i]:origColors[i]);
  m.geometry.attributes.color.needsUpdate=true;});renderer.shadowMap.needsUpdate=true;}
function setShade(on){useShade=on;key.castShadow=on;if(ground)ground.visible=on;
  renderer.shadowMap.autoUpdate=false;if(on)renderer.shadowMap.needsUpdate=true;}
function resize(){var W=vp.clientWidth||innerWidth,H=vp.clientHeight||(innerHeight-60);
  renderer.setSize(W,H);cam.aspect=W/H;cam.updateProjectionMatrix();}
addEventListener("resize",resize);resize();
document.getElementById("colorToggle").addEventListener("click",function(e){
  if(e.target.classList.contains("tbtn")){this.querySelectorAll(".tbtn").forEach(function(b){b.classList.remove("active");});
    e.target.classList.add("active");setColorMode(e.target.getAttribute("data-mode"));}});
document.getElementById("shadeToggle").addEventListener("click",function(e){
  if(e.target.classList.contains("tbtn")){this.querySelectorAll(".tbtn").forEach(function(b){b.classList.remove("active");});
    e.target.classList.add("active");setShade(e.target.getAttribute("data-shade")==="on");}});
function setupModel(model){
  model.traverse(function(c){if(c.isMesh){meshes.push(c);
    var ca=c.geometry.attributes.color;origColors.push(ca);grayColors.push(buildGray(ca));
    c.material=new THREE.MeshLambertMaterial({vertexColors:true,side:THREE.FrontSide});
    c.castShadow=true;c.receiveShadow=false;}});
  scene.add(model);
  var box=new THREE.Box3().setFromObject(model),size=box.getSize(new THREE.Vector3()),ctr=box.getCenter(new THREE.Vector3());
  var maxD=Math.max(size.x,size.y,size.z)*1.15;
  key.position.set(ctr.x+maxD*0.55,ctr.y+maxD*1.3,ctr.z+maxD*0.5);key.target.position.copy(ctr);scene.add(key.target);
  key.shadow.mapSize.width=4096;key.shadow.mapSize.height=4096;
  var sc=key.shadow.camera;sc.left=-maxD;sc.right=maxD;sc.top=maxD;sc.bottom=-maxD;
  sc.near=maxD*0.05;sc.far=maxD*5;sc.updateProjectionMatrix();
  key.shadow.bias=-0.0003;key.shadow.normalBias=0;
  var gg=new THREE.PlaneGeometry(maxD*6,maxD*6),gm=new THREE.ShadowMaterial({opacity:0.14});
  ground=new THREE.Mesh(gg,gm);ground.rotation.x=-Math.PI/2;ground.position.y=box.min.y-size.y*0.002;
  ground.receiveShadow=true;scene.add(ground);
  cam.position.set(maxD*1.5,maxD*1.0,maxD*2.0);cam.near=maxD*0.002;cam.far=maxD*20;cam.updateProjectionMatrix();
  ctrl.target.copy(ctr);ctrl.minDistance=maxD*0.008;ctrl.maxDistance=maxD*8;ctrl.update();
  setShade(true);
  document.getElementById("ldg").classList.add("hide");
}
setTimeout(function(){try{lt.textContent="Modell wird entpackt…";
  var bin=atob(GLB_B64),bytes=new Uint8Array(bin.length);
  for(var i=0;i<bin.length;i++)bytes[i]=bin.charCodeAt(i);
  MeshoptDecoder.ready.then(function(){lt.textContent="Modell wird aufgebaut…";
    var loader=new THREE.GLTFLoader();loader.setMeshoptDecoder(MeshoptDecoder);
    loader.parse(bytes.buffer,"",function(gltf){setupModel(gltf.scene);},
      function(err){lt.textContent="Fehler: "+(err&&err.message?err.message:err);});
  }).catch(function(e){lt.textContent="Decoder-Fehler: "+e.message;});
}catch(e){lt.textContent="Fehler: "+e.message;}},100);
(function loop(){requestAnimationFrame(loop);ctrl.update();renderer.render(scene,cam);})();
</script>
</body></html>'''


# ════════════════════════════════════════════════════════════════════
#  Hauptprogramm
# ════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="STEP-Datei (.stp/.step)")
    ap.add_argument("--output", required=True, help="Ausgabe-HTML")
    ap.add_argument("--assets-dir", default="assets", help="Ordner mit Three.js-JS-Dateien")
    ap.add_argument("--model-name", default="Modell", help="Projektname (in der Fußzeile)")
    ap.add_argument("--expiry-days", type=int, default=20, help="Gültigkeit in Tagen")
    ap.add_argument("--gltfpack", default="gltfpack", help="Pfad zum gltfpack-Binary")
    args = ap.parse_args()

    expiry = (datetime.datetime.utcnow() + datetime.timedelta(days=args.expiry_days))
    expiry_iso = expiry.strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"Gokoba-Viewer-Konvertierung: {os.path.basename(args.input)}")
    print(f"  Gültig bis: {expiry_iso} ({args.expiry_days} Tage)")

    with tempfile.TemporaryDirectory() as tmp:
        raw_glb = os.path.join(tmp, "raw.glb")
        small_glb = os.path.join(tmp, "small.glb")
        print("[1/4] Meshing + Farben (OpenCascade)…")
        v, f, c = mesh_with_colors(args.input)
        print("[2/4] Normalen + Farb-Boost + GLB…")
        build_glb(v, f, c, raw_glb)
        print("[3/4] Komprimierung (gltfpack)…")
        compress_glb(raw_glb, small_glb, gltfpack=args.gltfpack)
        print("[4/4] HTML-Viewer erzeugen…")
        build_html(small_glb, args.assets_dir, args.model_name, expiry_iso, args.output)
    print(f"FERTIG → {args.output}")


if __name__ == "__main__":
    main()
