#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Gokoba 3D-Viewer, IFC-Zweig: IFC (+ optional EM.11) -> interaktive HTML-Ansicht mit Klick-Info.
Laeuft im GitHub-Workflow convert-ifc.yml. Benoetigt: pip install ifcopenshell trimesh numpy
und gltfpack (npm i -g gltfpack@1.1.0). viewer-template-ifc.html muss neben dieser Datei liegen.

Aufruf: python convert_ifc.py --input-dir jobs-ifc/<id> --output docs/v/<id>/index.html
        --model-name "Name" --expiry-days 40 --gltfpack gltfpack

Erkennung im Eingangsordner: alle *.ifc werden eingelesen; die EM.11-Datei wird am
Dateikopf erkannt (ViewDefinition SteelFabricationView), nicht am Namen. Ohne EM.11
laeuft die Wandlung normal, nur ohne Gelaender-Positionsnummern.
"""
import sys, os, json, re
import numpy as np
import ifcopenshell, ifcopenshell.geom
import ifcopenshell.util.element as ue
import trimesh

# ★ Farbtabelle: AS-Layername → RGB. AS schreibt Umlaute als '?' in die IFC,
#   deshalb stehen die Namen hier genauso. Bei Bedarf anpassen/ergaenzen.
LAYER_FARBE = {
    # ★ Verbindliche Gokoba-Layerfarben (aus dem AS-Layermanager, ACI -> RGB).
    #   AS schreibt Umlaute als '?' in die IFC - Namen deshalb genauso.
    'AS_Tr?ger':      (0x7F,0x7F,0xFF),  # ACI 171
    'AS_St?tze':      (0x7F,0x7F,0xFF),  # ACI 171
    'AS_Stuetzen':    (0x7F,0x7F,0xFF),  # ACI 171
    'AS_Treppe':      (0x7F,0x7F,0xFF),  # ACI 171
    'AS_Kanttr?ger':  (0x7F,0x7F,0xFF),  # ACI 171
    'AS_Bleche':      (0x00,0x3F,0x7F),  # ACI 154
    'AS_Fachwerk':    (0x00,0x3F,0x7F),  # ACI 154
    'AS_Gelaender':   (0xFF,0xBF,0x7F),  # ACI 31
    'AS_Gel?nder':    (0xFF,0xBF,0x7F),  # ACI 31
    'AS_Gitterroste': (0x5F,0x7F,0x3F),  # ACI 75
    'AS_Stufen':      (0x5F,0x7F,0x3F),  # ACI 75
    'AS_Schrauben':   (0x99,0x99,0x99),  # ACI 253
    'AS_Kopfbolzen':  (0x99,0x99,0x99),  # ACI 253
    'AS_Betondecken':    (0x7F,0x3F,0x3F),  # ACI 15
    'AS_Betonfundament': (0x7F,0x3F,0x3F),  # ACI 15
    'AS_Betontr?ger':    (0x7F,0x3F,0x3F),  # ACI 15
    'AS_Betonw?nde':     (0x7F,0x3F,0x3F),  # ACI 15
    'AS_Sonderteile': (0xA5,0x67,0x52),  # ACI 23
    'AS_Ankerk?fige': (0xA5,0x67,0x52),  # ACI 23
    'AS_Kameras':     (0xA5,0x67,0x52),  # ACI 23
    'AS_Pfette':      (0x00,0xA5,0x7C),  # ACI 122
    'AS_Verband':     (0x7F,0x1F,0x00),  # ACI 24
    'AS_Schwei?en':   (0x00,0x7C,0xA5),  # ACI 142
    'AS_Raster':      (0x39,0x4C,0x26),  # ACI 77
    'AS_Fassade':     (0x00,0x7F,0x00),  # ACI 94
    'AS_Holztr?ger':  (0x7F,0x3F,0x00),  # ACI 34
    'AS_Knoten':      (0xFF,0x00,0xFF),  # ACI 210
    'AS_H?henkoten':  (0xFF,0x00,0x00),  # ACI 1
    'AS_Arbeitsfl?chen':        (0xFF,0x00,0x3F),  # ACI 240
    'AS_Standard':              (0xE8,0xE8,0xE8),  # ACI 7 (weiss, leicht gedimmt fuer den Viewer)
    'AS_Anschlussboxen':        (0xE8,0xE8,0xE8),  # ACI 7
    'AS_Strukturelementrahmen': (0xE8,0xE8,0xE8),  # ACI 7
    'A_BWS':        (0x4C,0x00,0x4C),  # ACI 216
    'A_D?mmung':    (0x3F,0x4F,0x7F),  # ACI 165
    'A_Edelstahl':  (0xFF,0x7F,0x00),  # ACI 30
    'A_Estrich':    (0x7F,0x3F,0x7F),  # ACI 215
    'A_Glas':       (0xFF,0xFF,0x7F),  # ACI 51
    'A_Mauerwerk':  (0x26,0x4C,0x2F),  # ACI 107
    'A_Mineralit':  (0x7F,0x00,0x1F),  # ACI 244
    'A_Riffelblech':(0x58,0x13,0x17),  # ACI 249
    'A_Trespa_1':   (0x13,0x00,0x4C),  # ACI 186
    'A_Trespa_2':   (0x67,0x52,0xA5),  # ACI 183
    'A_Trespa_3':   (0x4C,0x26,0x00),  # ACI 36
    'A_Trespa_4':   (0x26,0x13,0x00),  # ACI 38
}

STANDARD_FARBE = (0x74, 0x6A, 0x5C)   # alles ohne Layerzuordnung (z.B. Sonderteile)

# ★ Layer -> ACI-Nummer (aus dem Viewer-Template GOKOBA_LAYER_ACI; '?' = kaputter AS-Umlaut).
#   Der Viewer erkennt die Nummer am Materialnamen GOKOBA_ACI_<n>_... und wendet sein
#   verbindliches Anzeige-Farbschema an. Unbekannte Layer -> 0 (Standard dunkles Blau).
LAYER_ACI = {
    'AS_Tr?ger':171,'AS_St?tze':171,'AS_Stuetzen':171,'AS_Treppe':171,'AS_Kanttr?ger':171,
    'AS_Bleche':81,'AS_Fachwerk':154,'Gokoba_Quertr?ger':154,
    'AS_Gelaender':145,'AS_Gel?nder':145,
    'AS_Gitterroste':75,'AS_Stufen':75,
    'AS_Schrauben':253,'AS_Kopfbolzen':253,'Gokoba_Bestand':253,
    'AS_Betondecken':15,'AS_Betonfundament':15,'AS_Betontr?ger':15,'AS_Betonw?nde':15,
    'AS_Sonderteile':23,'AS_Ankerk?fige':23,'AS_Kameras':23,
    'AS_Pfette':122,'AS_Verband':24,'AS_Schwei?en':142,'AS_Raster':77,'AS_Fassade':94,
    'AS_Holztr?ger':34,'AS_Knoten':210,'AS_H?henkoten':1,'AS_Arbeitsfl?chen':240,
    'AS_Standard':7,'AS_Anschlussboxen':7,'AS_Strukturelementrahmen':7,'0':7,'Advance_DefaultLayer':7,
    'A_D?mmung':165,'Gokoba_Einfassleisten':165,'A_Estrich':215,'A_Mauerwerk':107,
    'A_Edelstahl':30,'A_Glas':51,'A_Riffelblech':249,'A_BWS':216,'A_Mineralit':244,
    'A_Trespa_1':186,'A_Trespa_2':183,'A_Trespa_3':36,'A_Trespa_4':38,
    'Cold rolled':254,'Gokoba_Alu-Wannen':254,'Gokoba_BWS':99,'Nicht_ableiten':31,
}
DICHTE_STAHL = 7850.0                 # kg/m3 fuer die Gewichtsschaetzung
SCHRAUBEN_TYPEN = ('IfcFastener', 'IfcMechanicalFastener')
BETON_LAYER = ('AS_Betondecken', 'AS_Betonw?nde')


# ★ Bauteilart aus Layer + IFC-Typ (Grundlage der Kaertchen-Gestaltung im Viewer)
ART_LAYER = {
    'AS_Tr?ger':'profil','AS_St?tze':'profil','AS_Stuetzen':'profil','AS_Treppe':'profil',
    'AS_Pfette':'profil','AS_Verband':'profil','AS_Fachwerk':'profil',
    'AS_Kanttr?ger':'kantprofil',
    'AS_Bleche':'blech',
    'AS_Gitterroste':'gitterrost','AS_Stufen':'gitterroststufe',
    'AS_Schrauben':'schraube','AS_Kopfbolzen':'kopfbolzen','AS_Ankerk?fige':'anker',
    'AS_Sonderteile':'sonderteil',
    'AS_Betondecken':'beton','AS_Betonw?nde':'beton','AS_Betonfundament':'beton','AS_Betontr?ger':'beton',
}
def art_von(layer, typ):
    a = ART_LAYER.get(layer)
    if a: return a
    if typ == 'IfcPlate': return 'blech'
    if typ == 'IfcMechanicalFastener': return 'schraube'
    if typ == 'IfcFastener': return 'schraube'
    if typ == 'IfcBuildingElementProxy': return 'sonderteil'
    if typ in ('IfcWall','IfcSlab'): return 'beton'
    if typ in ('IfcBeam','IfcColumn','IfcMember'): return 'profil'
    return 'sonstiges'

def masse_aus_obb(m):
    """Orientierte Box: liefert (a, b, dicke) in mm, absteigend sortiert, oder None."""
    try:
        ext = sorted(float(x) * 1000.0 for x in m.bounding_box_oriented.primitive.extents)
        return (ext[2], ext[1], ext[0])
    except Exception:
        try:
            ext = sorted(float(x) * 1000.0 for x in m.extents)
            return (ext[2], ext[1], ext[0])
        except Exception:
            return None

def ist_em11(pfad):
    try:
        with open(pfad, 'rb') as f:
            return b'SteelFabricationView' in f.read(600)
    except Exception:
        return False

def wandle(ifc_pfad, em11_pfad, ohne_schrauben=False, ohne_beton=False):
    """Kern aus ifc2glb v4: liefert (glb_pfad, teile_dict) im Ordner der IFC."""
    import time, json as _json
    basisname = re.sub(r'\.ifc$', '', ifc_pfad, flags=re.I)
    t0 = time.time()
    f = ifcopenshell.open(ifc_pfad)
    print('* IFC geladen: %s (%s)' % (ifc_pfad, f.schema))

    lay = {}
    for la in f.by_type('IfcPresentationLayerAssignment'):
        for it in la.AssignedItems:
            lay[it.id()] = la.Name
    def layer_von(prod):
        try:
            rep = prod.Representation
            if not rep: return None
            for r in rep.Representations:
                if r.id() in lay: return lay[r.id()]
                for it in r.Items:
                    if it.id() in lay: return lay[it.id()]
        except Exception:
            pass
        return None

    def daten_von(prod):
        d = {'ref': None, 'profil': None, 'familie': None, 'material': None, 'laenge': None}
        try:
            for pn, props in ue.get_psets(prod).items():
                if pn == 'ProfileProperties':
                    d['profil'] = props.get('Section')
                    d['familie'] = props.get('SectionFamily')
                else:
                    r = props.get('Reference')
                    if r and str(r) != 'nicht definiert':
                        d['ref'] = str(r)
                    if props.get('Span'):
                        d['laenge'] = round(float(props['Span']), 1)
        except Exception:
            pass
        try:
            for rel in getattr(prod, 'HasAssociations', []) or []:
                if rel.is_a('IfcRelAssociatesMaterial'):
                    m = rel.RelatingMaterial
                    nm = getattr(m, 'Name', None)
                    if nm: d['material'] = nm
        except Exception:
            pass
        return d

    em_zentren = None
    if em11_pfad:
        print('* Lese EM.11 fuer Positionsnummern: ' + em11_pfad)
        f2 = ifcopenshell.open(em11_pfad)
        def mark_von(e):
            try:
                for pn, props in ue.get_psets(e).items():
                    m = props.get('PieceMark') or props.get('Reference')
                    if m and str(m) not in ('nicht definiert', ''): return str(m)
            except Exception:
                pass
            return None
        s2 = ifcopenshell.geom.settings(); s2.set(s2.USE_WORLD_COORDS, True)
        it2 = ifcopenshell.geom.iterator(s2, f2, 4)
        pkt = []; mrk = []
        if it2.initialize():
            while True:
                try:
                    shp2 = it2.get(); p2 = f2.by_id(shp2.id)
                    if p2.is_a() != 'IfcOpeningElement':
                        m = mark_von(p2)
                        if m:
                            v2 = np.array(shp2.geometry.verts).reshape(-1, 3)
                            pkt.append(v2.mean(axis=0)); mrk.append(m)
                except Exception:
                    pass
                if not it2.next(): break
        if pkt:
            em_zentren = (np.array(pkt), mrk)
            print('  EM.11: %d gekennzeichnete Teile' % len(mrk))

    s = ifcopenshell.geom.settings()
    s.set(s.USE_WORLD_COORDS, True)
    it = ifcopenshell.geom.iterator(s, f, 4)
    szene = trimesh.Scene()
    material_cache = {}
    teile = {}
    n = 0; fehler = 0

    def material_fuer(layer):
        if layer in material_cache: return material_cache[layer]
        col = LAYER_FARBE.get(layer, STANDARD_FARBE)
        mat = trimesh.visual.material.PBRMaterial(
            baseColorFactor=[col[0]/255.0, col[1]/255.0, col[2]/255.0, 1.0],
            metallicFactor=0.15, roughnessFactor=0.7)
        aci = LAYER_ACI.get(layer, 0)
        mat.name = 'GOKOBA_ACI_%d_%s' % (aci, re.sub(r'[^A-Za-z0-9_]', '_', str(layer)))
        material_cache[layer] = mat
        return mat

    if not it.initialize():
        raise SystemExit('Geometrie-Iterator konnte nicht starten.')
    while True:
        try:
            shp = it.get()
            prod = f.by_id(shp.id)
            typ = prod.is_a()
            if typ != 'IfcOpeningElement':
                L = layer_von(prod)
                if (ohne_schrauben and typ in SCHRAUBEN_TYPEN) or (ohne_beton and L in BETON_LAYER):
                    pass
                else:
                    g = shp.geometry
                    v = np.array(g.verts).reshape(-1, 3)
                    fc = np.array(g.faces).reshape(-1, 3)
                    m = trimesh.Trimesh(vertices=v, faces=fc, process=False)
                    m.visual = trimesh.visual.TextureVisuals(material=material_fuer(L))
                    kn = 'E%d' % shp.id
                    szene.add_geometry(m, node_name=kn, geom_name=kn)
                    d = daten_von(prod)
                    d['typ'] = typ
                    d['layer'] = L
                    d['art'] = art_von(L, typ)
                    if typ == 'IfcMechanicalFastener':
                        # ★ Groesse steckt in den Klassik-Attributen, nicht in Psets
                        try:
                            dm = getattr(prod, 'NominalDiameter', None)
                            ln = getattr(prod, 'NominalLength', None)
                            ot = (getattr(prod, 'ObjectType', '') or '')
                            if 'Anchor' in ot or L == 'AS_Ankerk?fige':
                                d['art'] = 'anker'
                            if dm and ln:
                                fmt = lambda x: ('%g' % round(float(x), 1))
                                vor = '\u00f8' if d['art'] == 'anker' else 'M'
                                d['groesse'] = vor + fmt(dm) + ' x ' + fmt(ln)
                        except Exception:
                            pass
                    # ★ Seitliche Laschen/Stäbe auf Rost-/Stufenlayern tragen ein Profil -> Profilkarte
                    if d['art'] in ('gitterrost', 'gitterroststufe') and d.get('profil'):
                        d['art'] = 'profil'
                    if d['art'] in ('blech', 'kantblech', 'gitterrost', 'gitterroststufe'):
                        mm = masse_aus_obb(m)
                        if mm:
                            d['masse'] = [round(mm[0]), round(mm[1]), round(mm[2], 1)]
                    if d['art'] in ('profil', 'kantprofil') and not d.get('laenge'):
                        mm = masse_aus_obb(m)
                        if mm:
                            d['laenge'] = round(mm[0], 1)   # laengste Kante = Stablaenge
                    if d['ref'] is None and em_zentren is not None:
                        z = v.mean(axis=0)
                        dist = np.linalg.norm(em_zentren[0] - z, axis=1)
                        k = int(dist.argmin())
                        if dist[k] < 0.005:
                            d['ref'] = em_zentren[1][k]
                    try:
                        vol = float(m.volume)
                        if vol > 0 and (L not in BETON_LAYER) and d['art'] not in ('schraube', 'anker'):
                            d['gewicht'] = round(vol * DICHTE_STAHL, 1)
                    except Exception:
                        pass
                    teile[kn] = d
                    n += 1
        except Exception:
            fehler += 1
        if not it.next():
            break
    print('* Vermascht: %d Bauteile | Fehler: %d | %.0fs' % (n, fehler, time.time() - t0))
    if n == 0:
        raise SystemExit('Keine Bauteile in der IFC.')
    glb = basisname + '.glb'
    szene.export(glb)
    return glb, teile

VORLAGE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'viewer-template-ifc.html')

def main():
    import argparse, base64, datetime, subprocess, tempfile, glob
    ap = argparse.ArgumentParser()
    ap.add_argument('--input-dir', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--model-name', default='Modell')
    ap.add_argument('--expiry-days', type=int, default=40)
    ap.add_argument('--gltfpack', default='gltfpack')
    ap.add_argument('--ohne-schrauben', action='store_true')
    ap.add_argument('--ohne-beton', action='store_true')
    args = ap.parse_args()

    ifcs = sorted(glob.glob(os.path.join(args.input_dir, '*.ifc')))
    haupt = [p for p in ifcs if not ist_em11(p)]
    em11 = [p for p in ifcs if ist_em11(p)]
    if not haupt:
        raise SystemExit('Keine IFC2x3-Datei (CoordinationView) im Ordner gefunden.' +
                         (' Nur EM.11 vorhanden - bitte auch den normalen IFC2x3-Export beilegen.' if em11 else ''))
    ifc = haupt[0]
    em = em11[0] if em11 else None
    if not em:
        print('Hinweis: keine EM.11-Datei dabei - Gelaenderteile bekommen keine Positionsnummern.')

    glb, teile = wandle(ifc, em, args.ohne_schrauben, args.ohne_beton)
    small = glb.replace('.glb', '_pack.glb')
    print('* Komprimierung (gltfpack -cc -kn -km -noq) ...')
    subprocess.run([args.gltfpack, '-i', glb, '-o', small, '-cc', '-kn', '-km', '-noq'], check=True)

    print('* HTML-Ansicht erzeugen ...')
    html = open(VORLAGE, encoding='utf-8').read()
    g64 = base64.b64encode(open(small, 'rb').read()).decode('ascii')
    t64 = base64.b64encode(json.dumps(teile, ensure_ascii=False, separators=(',', ':')).encode('utf-8')).decode('ascii')
    expiry = (datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=args.expiry_days)).strftime('%Y-%m-%dT%H:%M:%SZ')
    html = html.replace('__GLB_B64__', g64)
    html = html.replace('__TEILE_B64__', t64)
    html = html.replace('__PROJ_NAME__', args.model_name)
    html = html.replace('__EXPIRY__', expiry)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    open(args.output, 'w', encoding='utf-8').write(html)
    print('OK: ' + args.output + ' (%d KB)' % (os.path.getsize(args.output) // 1024))

if __name__ == '__main__':
    main()
