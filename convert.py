#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gokoba 3D-Viewer - Konvertierungs-Pipeline
===========================================
Wandelt eine Advance-Steel-STEP-Datei in einen eigenständigen Web-3D-Viewer (HTML) um.

Ablauf:
  STEP  →  OpenCascade (Meshing + Farben pro Bauteil, sRGB)
        →  crease-angle Normalen (40°, scharfe Kanten / glatte Rundungen)
        →  Farb-Boost (HSL S*1.12, dann L*0.78 gesamt - Gokoba-Look)
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
<meta name="color-scheme" content="light">
<title>Gokoba 3D-Viewer, __MODEL_NAME__</title>
<link rel="icon" type="image/png" sizes="32x32" href="data:image/png;base64,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">
<link rel="apple-touch-icon" sizes="192x192" href="data:image/png;base64,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">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Hanken+Grotesk:wght@400;500;600;700;800&family=Newsreader:opsz,wght@16..72,400;16..72,500;16..72,600&display=swap" rel="stylesheet">
<style>
*{box-sizing:border-box;margin:0;padding:0;}
:root{color-scheme:light;--sand:#F4F1EA;--paper2:#ECE6DA;--acc:#B05828;--acc-dk:#8E4319;--acc-lt:#E59A63;--ink:#1A1813;--ink2:#6E665A;--brd:#E3DDCE;--serif:'Newsreader',Georgia,serif;--display:'Hanken Grotesk',system-ui,sans-serif;--foot1:#33271A;--foot2:#20180F;--foot-tx:#C9C0B0;--foot-tx2:#8B8170;}
body{font-family:var(--display);background:var(--sand);height:100vh;height:100dvh;overflow:hidden;}
header{position:fixed;top:0;left:0;right:0;height:60px;z-index:100;background:#fff;
  border-bottom:1px solid var(--brd);display:flex;align-items:center;justify-content:center;padding:0 22px;}
.header-inner{width:100%;max-width:820px;display:flex;align-items:center;justify-content:space-between;gap:16px;}
.brand{display:flex;align-items:center;gap:13px;min-width:0;}
.logo{display:flex;align-items:center;gap:11px;text-decoration:none;flex:none;}
.logo .mark{width:34px;height:34px;flex:none;}
.logo .mark svg{width:100%;height:100%;display:block;}
.logo .wm{font-family:var(--serif);font-weight:600;font-size:1.5rem;letter-spacing:-.012em;color:var(--ink);line-height:1;}
.vsep{width:1px;height:30px;background:var(--brd);flex:none;}
.proj{display:flex;flex-direction:column;gap:1px;min-width:0;}
.proj-tag{font-size:9px;font-weight:600;letter-spacing:.14em;text-transform:uppercase;color:var(--ink2);}
.proj-name{font-family:var(--serif);font-weight:600;font-size:1.12rem;color:var(--acc);line-height:1.15;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:40vw;}
.controls{display:flex;gap:10px;align-items:center;flex-wrap:wrap;justify-content:flex-end;}
.toggle{display:flex;border:1px solid var(--brd);border-radius:8px;overflow:hidden;}
.tbtn{font-family:inherit;font-size:11px;font-weight:500;padding:7px 14px;
  border:none;background:#fff;color:var(--ink2);cursor:pointer;transition:all .15s;white-space:nowrap;}
.tbtn.active{background:var(--acc);color:#fff;}
.tbtn:not(.active):hover{background:var(--paper2);color:var(--ink);}
#vpanel{height:100vh;height:100dvh;padding-top:60px;position:relative;}
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
#bar{position:absolute;bottom:0;left:0;right:0;background:linear-gradient(140deg,var(--foot1),var(--foot2));
  border-top:1px solid rgba(244,241,234,.10);color:var(--foot-tx);padding:11px 22px;
  padding-bottom:calc(11px + env(safe-area-inset-bottom,0px));display:flex;justify-content:center;}
.bar-inner{width:100%;max-width:1000px;display:flex;align-items:center;justify-content:space-between;gap:16px;font-size:.82rem;}
.fl{display:flex;align-items:center;gap:10px;min-width:0;}
.fmark{width:24px;height:24px;flex:none;}.fmark svg{width:100%;height:100%;display:block;}
.fwm{font-family:var(--serif);font-weight:600;font-size:1.06rem;color:#F4F1EA;letter-spacing:-.01em;}
.cpr{color:var(--foot-tx2);font-size:.8rem;white-space:nowrap;}
.cpr a{color:var(--foot-tx2);text-decoration:none;border-bottom:1px solid transparent;transition:color .18s,border-color .18s;}
.cpr a:hover{color:var(--acc-lt);border-color:var(--acc-lt);}
.fc{color:var(--foot-tx);font-size:.8rem;text-align:center;white-space:nowrap;}
.fr{display:flex;align-items:center;gap:14px;min-width:0;}
.fweb{color:var(--foot-tx);text-decoration:none;font-weight:500;transition:color .18s;white-space:nowrap;}
.fweb:hover{color:var(--acc-lt);}
.hint{font-size:.72rem;color:var(--foot-tx2);opacity:.85;white-space:nowrap;}
@media (max-width:600px){
  body{display:flex;flex-direction:column;}
  header{position:static;height:auto;padding:8px 12px;}
  .header-inner{flex-wrap:wrap;justify-content:center;gap:8px 12px;}
  .brand{gap:10px;}
  .logo .wm{font-size:1.3rem;}
  .proj-name{max-width:52vw;font-size:1rem;}
  .controls{width:100%;justify-content:center;gap:8px;}
  .tbtn{padding:6px 12px;font-size:10px;}
  #vpanel{flex:1;height:auto;padding-top:0;}
  #bar{padding:8px 14px;padding-bottom:calc(8px + env(safe-area-inset-bottom,0px));}
  .bar-inner{gap:10px;flex-wrap:wrap;justify-content:center;text-align:center;}
  .fc,.hint{display:none;}
  .cpr{white-space:normal;}
  .fwm{font-size:1rem;}
}
</style>
</head>
<body>
<header>
  <div class="header-inner">
    <div class="brand">
      <a class="logo" href="https://www.gokoba.com" target="_blank" rel="noopener" aria-label="Gokoba, zur Website">
        <span class="mark"><svg viewBox="0 0 100 100"><defs><linearGradient id="gkG_hdr" x1="0" y1="0" x2="0" y2="100" gradientUnits="userSpaceOnUse"><stop offset="0" stop-color="#BD6D40"/><stop offset="0.5" stop-color="#AC5A2C"/><stop offset="1" stop-color="#9C4B1D"/></linearGradient><filter id="gkS_hdr" x="-30%" y="-30%" width="160%" height="170%"><feDropShadow dx="1" dy="2.4" stdDeviation="2.2" flood-color="#331A06" flood-opacity="0.34"/></filter></defs><rect width="100" height="100" rx="23" fill="url(#gkG_hdr)"/><path d="M13.6 76 L13.6 63 L31.8 63 L31.8 50 L50 50 L50 37 L68.2 37 L68.2 24 L86.4 24 L86.4 41 Z" fill="#F5F2EB" filter="url(#gkS_hdr)"/></svg></span>
        <span class="wm">Gokoba</span>
      </a>
      <span class="vsep"></span>
      <div class="proj"><span class="proj-tag">3D-Viewer</span><span class="proj-name">__MODEL_NAME__</span></div>
    </div>
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
      <div class="fl">
        <span class="fmark"><svg viewBox="0 0 100 100"><defs><linearGradient id="gkG_ftr" x1="0" y1="0" x2="0" y2="100" gradientUnits="userSpaceOnUse"><stop offset="0" stop-color="#BD6D40"/><stop offset="0.5" stop-color="#AC5A2C"/><stop offset="1" stop-color="#9C4B1D"/></linearGradient><filter id="gkS_ftr" x="-30%" y="-30%" width="160%" height="170%"><feDropShadow dx="1" dy="2.4" stdDeviation="2.2" flood-color="#331A06" flood-opacity="0.34"/></filter></defs><rect width="100" height="100" rx="23" fill="url(#gkG_ftr)"/><path d="M13.6 76 L13.6 63 L31.8 63 L31.8 50 L50 50 L50 37 L68.2 37 L68.2 24 L86.4 24 L86.4 41 Z" fill="#F5F2EB" filter="url(#gkS_ftr)"/></svg></span>
        <span class="fwm">Gokoba</span>
        <span class="cpr">&copy; 2026 <a href="mailto:info@gokoba.com">Paul Thomas</a> &middot; <a href="mailto:info@gokoba.com">info@gokoba.com</a> &middot; <a href="https://www.gokoba.com" target="_blank" rel="noopener">www.gokoba.com</a></span>
      </div>
      <div class="fr">
        <span class="hint">Ziehen: drehen &middot; Scroll: zoomen &middot; Rechtsklick: verschieben</span>
      </div>
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
