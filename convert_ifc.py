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
KONFIG = {}

def saniere_profil(p):
    if p and ('Autodesk' in str(p) or 'ProfileType' in str(p)): return None
    """AS-Profilname wie 'HEB DIN18800-1#@§@#HEB140' -> 'HEB 140'; 'FL60X10' -> 'FL 60x10'."""
    if not p: return p
    import re as _re
    s2 = str(p).split('#@§@#')[-1].strip()
    s2 = _re.sub(r'^([A-Za-z]+)\s*(\d)', r'\1 \2', s2)
    s2 = _re.sub(r'(\d)\s*[xX*]\s*(\d)', r'\1x\2', s2)
    return s2

def _knick_normalen(m, winkel_grad=26.0, fl_id=None, fl_breit=None):
    """v56: Mitteln nur schmal<->schmal (Radienstreifen, Rohrsegmente, bis winkel_grad)
    und fast-koplanare Naehte (<8 Grad). Breite Ebenen (>12mm Streifenbreite) bleiben
    STRIKT flach - Winkel allein kann Radius<->Ebene (~11 Grad) nie von Rohrsegmenten
    (~15 Grad) trennen, die Flaechenbreite kann es."""
    try:
        import collections
        F = m.faces; V = m.vertices; fn = m.face_normals
        if len(F) == 0: return m
        cosw = np.cos(np.radians(winkel_grad))
        cos8 = np.cos(np.radians(8.0))
        hatB = fl_id is not None and fl_breit is not None and len(fl_id) == len(F)
        vf = collections.defaultdict(list)
        for fi in range(len(F)):
            f = F[fi]
            vf[f[0]].append(fi); vf[f[1]].append(fi); vf[f[2]].append(fi)
        NV = np.zeros((len(F) * 3, 3)); PV = np.zeros((len(F) * 3, 3))
        for fi in range(len(F)):
            f = F[fi]
            for ci in range(3):
                v = f[ci]; n = fn[fi].copy()
                for gi in vf[v]:
                    if gi == fi: continue
                    dv = float(np.dot(fn[fi], fn[gi]))
                    if hatB:
                        bi = bool(fl_breit[fl_id[fi]]); bj = bool(fl_breit[fl_id[gi]])
                        grenze = cosw if (not bi and not bj) else cos8
                    else:
                        grenze = cosw
                    if dv > grenze:
                        n = n + fn[gi]
                l = float(np.linalg.norm(n))
                NV[fi * 3 + ci] = n / l if l > 1e-9 else fn[fi]
                PV[fi * 3 + ci] = V[v]
        m2 = trimesh.Trimesh(vertices=PV, faces=np.arange(len(F) * 3).reshape(-1, 3), process=False)
        m2.vertex_normals = NV
        return m2
    except Exception:
        return m

def norm_layer(l):
    """Direktweg liefert Layer mit ECHTEN Umlauten (AS_Traeger mit ae-Umlaut),
    die Tabellen-Schluessel stehen in AS-Exportschreibweise mit '?' -
    fuer den Abgleich werden Umlaute/ss auf '?' gefaltet."""
    import re as _re
    return _re.sub(u'[\u00e4\u00f6\u00fc\u00c4\u00d6\u00dc\u00df]', '?', l or '')

LAYER_FARBE = {
    # ★ Verbindliche Gokoba-Layerfarben (aus dem AS-Layermanager, ACI -> RGB).
    #   AS schreibt Umlaute als '?' in die IFC - Namen deshalb genauso.
    'AS_Tr?ger':      (0x3C,0x55,0xA8),  # kraeftiges dunkleres Blau (Pauls Wunsch)
    'AS_St?tze':      (0x3C,0x55,0xA8),
    'AS_Stuetzen':    (0x3C,0x55,0xA8),
    'AS_Treppe':      (0x3C,0x55,0xA8),
    'AS_Kanttr?ger':  (0x3C,0x55,0xA8),
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
    a = ART_LAYER.get(norm_layer(layer))
    if a: return a
    if typ == 'IfcPlate': return 'blech'
    if typ == 'IfcMechanicalFastener': return 'schraube'
    if typ == 'IfcDiscreteAccessory': return 'schraube'   # Muttern/Scheiben (EM.11)
    if typ == 'IfcFastener': return 'schraube'
    if typ == 'IfcBuildingElementProxy': return 'sonderteil'
    if typ in ('IfcWall','IfcSlab'): return 'beton'
    if typ in ('IfcBeam','IfcColumn','IfcMember'): return 'profil'
    return 'sonstiges'

def schraube_aus_symbol(f, prod):
    """EM.11-Schraube: Strichsymbol + Attribute -> einfacher Schaft mit Sechskantkopf."""
    try:
        import ifcopenshell.util.placement as _pl
        import ifcopenshell.util.unit as _un
        skal = _un.calculate_unit_scale(f)
        dm = float(getattr(prod, 'NominalDiameter', 0) or 0) * skal
        ln = float(getattr(prod, 'NominalLength', 0) or 0) * skal
        if dm <= 0: dm = 0.012
        if ln <= 0: ln = dm * 3.0
        M = _pl.get_local_placement(prod.ObjectPlacement)
        pA = pB = None
        for rep in (prod.Representation.Representations if prod.Representation else []):
            for it in rep.Items:
                if it.is_a('IfcPolyline') and len(it.Points) >= 2:
                    a = np.array(list(it.Points[0].Coordinates) + [0.0])[:3] * skal
                    b = np.array(list(it.Points[-1].Coordinates) + [0.0])[:3] * skal
                    pA = (M @ np.append(a, 1.0))[:3]
                    pB = (M @ np.append(b, 1.0))[:3]
                    break
            if pA is not None: break
        if pA is None:
            pA = M[:3, 3]
            pB = pA + M[:3, 2] * ln
        achse = pB - pA
        al = np.linalg.norm(achse)
        achse = achse / al if al > 1e-9 else M[:3, 2]
        def zyl(radius, hoehe, mitte):
            T = np.eye(4)
            z = np.array([0.0, 0.0, 1.0])
            v = np.cross(z, achse); c = float(np.dot(z, achse))
            if np.linalg.norm(v) > 1e-9:
                vx = np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])
                T[:3,:3] = np.eye(3) + vx + vx @ vx * (1.0/(1.0+c))
            elif c < 0:
                T[:3,:3] = np.diag([1.0,-1.0,-1.0])
            T[:3,3] = mitte
            return trimesh.creation.cylinder(radius=radius, height=hoehe, sections=6, transform=T)
        schaft = zyl(dm*0.5, ln, pA + achse*(ln*0.5))
        kopf = zyl(dm*0.95, dm*0.64, pA - achse*(dm*0.32))
        return trimesh.util.concatenate([kopf, schaft])
    except Exception:
        return None

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

def _pseudo(w):
    """AS-Platzhalter ('nicht definiert' u.ae.) als leer behandeln (v47)."""
    if not w: return None
    if str(w).strip().lower() in ('nicht definiert', 'not defined', '-', '?'): return None
    return w

def lese_namen(input_dir):
    """namen.txt aus dem Plugin: pos|klasse|x|y|z|name (Koordinaten in mm)."""
    import glob as _g
    p = _g.glob(os.path.join(input_dir, 'namen.txt'))
    je_pos = {}; je_ort = []; je_bg = {}
    if not p: return je_pos, je_ort, je_bg
    try:
        for zeile in open(p[0], encoding='utf-8', errors='replace'):
            t = zeile.rstrip('\n').split('|')
            if len(t) < 6: continue
            pos, klasse, x, y, z, name = t[0], t[1], t[2], t[3], t[4], t[5]
            pos = _pseudo(pos) or ''
            attrs = [w.strip() for w in t[6:11]]
            while len(attrs) < 5: attrs.append('')
            if klasse == 'Baugruppe':
                if pos and name.strip(): je_bg.setdefault(pos, name.strip())
                continue
            eintrag = {'klasse': klasse, 'name': name.strip(), 'attrs': attrs}
            if pos: je_pos.setdefault(pos, eintrag)
            if x and y and z:
                try:
                    je_ort.append((np.array([float(x), float(y), float(z)]) / 1000.0, eintrag))
                except Exception:
                    pass
    except Exception:
        pass
    return je_pos, je_ort, je_bg

def name_deute(d, name):
    """Objektname in Kaertchenfelder uebersetzen (x- und /-Schreibweisen)."""
    import re as _re
    d['name'] = name
    d['roh'] = name  # Diagnose: kompletter Objektname bleibt unsichtbar in den Daten
    unten = name.lower()
    if 'roststufe' in unten or 'stufe' in unten: d['art'] = 'gitterroststufe'
    elif 'rost' in unten or 'grating' in unten or 'graiting' in unten: d['art'] = 'gitterrost'
    if 'anker' in unten or 'hilti' in unten: d['art'] = 'anker'; d.setdefault('din', name)
    if d['art'] in ('gitterrost', 'gitterroststufe'):
        d['hersteller'] = name.split()[0] if name.split() else None
        paare = _re.findall(r'(\d+(?:[\.,]\d+)?)\s*[x/]\s*(\d+(?:[\.,]\d+)?)', name)
        if paare:
            d['masche'] = paare[0][0] + 'x' + paare[0][1]
        # Tragstab: bevorzugt das Paar nach '-' oder hinter MW/TS-Kuerzeln, sonst das zweite Paar
        m = _re.search(r'(?:-|TS\s*|Tragstab\s*)\s*(\d+(?:[\.,]\d+)?)\s*[x/]\s*(\d+(?:[\.,]\d+)?)', name, _re.I)
        if m:
            d['tragstab'] = m.group(1) + 'x' + m.group(2)
        elif len(paare) > 1:
            d['tragstab'] = paare[1][0] + 'x' + paare[1][1]
        if d.get('masche') and d.get('tragstab') == d.get('masche'):
            d['tragstab'] = None
        # Abm.AxB: A ist die Tragstabrichtung -> Masse entsprechend ordnen
        m = _re.search(r'Abm\.?\s*(\d+)\s*x\s*(\d+)', name, _re.I)
        if m and d.get('masse'):
            a, b = float(m.group(1)), float(m.group(2))
            dicke = d['masse'][2]
            d['masse'] = [round(a), round(b), dicke]
        # Beim Rost/Stufe den Klarnamen nicht doppelt im Kopf fuehren
        d['name'] = None
        d['hersteller'] = d['hersteller'] if d['hersteller'] and d['hersteller'][0].isupper() else None
        if d['hersteller'] and d['hersteller'].lower() in ('gitterrost', 'gitterroststufe', 'rost'):
            d['hersteller'] = None

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
                    d['profil'] = saniere_profil(props.get('Section'))
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
    em_bolzen = None
    em_baugruppen = None
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
        # ★ Schweissbaugruppen: Mitglied-Id -> Baugruppen-PieceMark
        bg_von = {}
        try:
            for rel in f2.by_type('IfcRelAggregates'):
                asm = rel.RelatingObject
                if not asm.is_a('IfcElementAssembly'):
                    continue
                bgm = mark_von(asm)
                if not bgm:
                    continue
                # ★ Instanz-Schluessel (jede Baugruppe einzeln) + Nummer fuer die Anzeige
                for mitglied in rel.RelatedObjects:
                    bg_von[mitglied.id()] = (str(asm.id()), bgm)
        except Exception:
            pass
        s2 = ifcopenshell.geom.settings(); s2.set(s2.USE_WORLD_COORDS, True)
        it2 = ifcopenshell.geom.iterator(s2, f2, 4)
        pkt = []; mrk = []; bgp = []; bgm2 = []
        if it2.initialize():
            while True:
                try:
                    shp2 = it2.get(); p2 = f2.by_id(shp2.id)
                    if p2.is_a() != 'IfcOpeningElement':
                        m = mark_von(p2)
                        bgx = bg_von.get(p2.id())
                        if m or bgx:
                            v2 = np.array(shp2.geometry.verts).reshape(-1, 3)
                            z2 = v2.mean(axis=0)
                            if m: pkt.append(z2); mrk.append(m)
                            if bgx: bgp.append(z2); bgm2.append(bgx)
                except Exception:
                    pass
                if not it2.next(): break
        if bgp:
            em_baugruppen = (np.array(bgp), bgm2)
            print('  EM.11: %d Teile mit Baugruppenzuordnung (%d Baugruppen)' % (len(bgm2), len(set(bgm2))))
        if pkt:
            em_zentren = (np.array(pkt), mrk)
            print('  EM.11: %d gekennzeichnete Teile' % len(mrk))
        # ★ Verbinder tragen in der EM.11 keine Geometrie - beide Exporte zaehlen die
        #   Bolzen aber in identischer Reihenfolge auf (am Testmodell 496/496 mit
        #   deckungsgleichen Massen). Abgleich daher ueber den Index, gesichert durch
        #   Durchmesser- und Laengenvergleich je Bolzen.
        try:
            em_liste = []
            for e2 in f2.by_type('IfcMechanicalFastener'):
                din = None; guete = None
                for pn, props in ue.get_psets(e2).items():
                    if pn == 'AISC_EM11_Pset_Bolt':
                        din = props.get('BoltType') or props.get('BoltStandard')
                        g = props.get('BoltGrade')
                        if isinstance(g, (list, tuple)): g = g[0] if g else None
                        guete = g
                em_liste.append((getattr(e2, 'NominalDiameter', None),
                                 getattr(e2, 'NominalLength', None), din, guete))
            if em_liste:
                em_bolzen = em_liste
                print('  EM.11: %d Verbinder mit DIN-Angaben' % sum(1 for x in em_liste if x[2]))
        except Exception:
            pass

    bolzen_index = {e.id(): i for i, e in enumerate(f.by_type('IfcMechanicalFastener'))}

    # ── Achsraster (IfcGrid) einsammeln: Linien + Beschriftung fuer den Viewer ──
    achsen = []
    try:
        import ifcopenshell.util.placement as up
        import ifcopenshell.util.unit as uu
        try:
            einheit = float(uu.calculate_unit_scale(f))  # Rohkoordinaten -> Meter (AS: 0.001)
        except Exception:
            einheit = 1.0
        for grid in f.by_type('IfcGrid'):
            try:
                M = up.get_local_placement(grid.ObjectPlacement)
            except Exception:
                M = np.eye(4)
            for richtung in (grid.UAxes or []), (grid.VAxes or []), (getattr(grid, 'WAxes', None) or []):
                for ax in richtung:
                    try:
                        kurve = ax.AxisCurve
                        pts = None
                        if kurve.is_a('IfcPolyline'):
                            pts = [p.Coordinates for p in kurve.Points]
                        elif kurve.is_a('IfcTrimmedCurve') and kurve.BasisCurve.is_a('IfcLine'):
                            continue  # getrimmte Linien sind selten; erstmal auslassen
                        if not pts or len(pts) < 2:
                            continue
                        p1 = np.array(list(pts[0]) + [0.0] * (3 - len(pts[0])), dtype=float)
                        p2 = np.array(list(pts[-1]) + [0.0] * (3 - len(pts[-1])), dtype=float)
                        w1 = (M @ np.append(p1, 1.0))[:3] * einheit
                        w2 = (M @ np.append(p2, 1.0))[:3] * einheit
                        achsen.append({'tag': str(ax.AxisTag or ''),
                                       'p': [round(float(x), 4) for x in list(w1) + list(w2)]})
                    except Exception:
                        continue
    except Exception:
        pass
    if achsen:
        print('* Achsraster: %d Achsen uebernommen' % len(achsen))

    s = ifcopenshell.geom.settings()
    s.set(s.USE_WORLD_COORDS, True)
    it = ifcopenshell.geom.iterator(s, f, 4)
    szene = trimesh.Scene()
    material_cache = {}
    teile = {}
    n = 0; fehler = 0

    ART_ERSATZ = {'profil': 'AS_Tr?ger', 'kantprofil': 'AS_Kanttr?ger', 'blech': 'AS_Bleche',
                  'schraube': 'AS_Schrauben', 'anker': 'AS_Ankerk?fige',
                  'gitterrost': 'AS_Gitterroste', 'gitterroststufe': 'AS_Stufen',
                  'sonderteil': 'AS_Sonderteile', 'beton': 'AS_Betondecken'}
    def material_fuer(layer, art=None):
        # ★ EM.11 kennt keine AS-Layer: dann bekommt das Teil die Farbe seiner Bauteilart
        if layer not in LAYER_FARBE and art in ART_ERSATZ:
            layer = ART_ERSATZ[art]
        if layer in material_cache: return material_cache[layer]
        col = LAYER_FARBE.get(norm_layer(layer), STANDARD_FARBE)
        mat = trimesh.visual.material.PBRMaterial(
            baseColorFactor=[col[0]/255.0, col[1]/255.0, col[2]/255.0, 1.0],
            metallicFactor=0.15, roughnessFactor=0.7)
        aci = LAYER_ACI.get(norm_layer(layer), 0)
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
                    m.visual = trimesh.visual.TextureVisuals(material=material_fuer(L, art_von(L, typ)))
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
                    if d['art'] in ('gitterrost', 'gitterroststufe') and d.get('masse'):
                        # ★ Flache Begleitteile (Stufenlaschen, Auflagerbleche) sind Bleche;
                        #   echte Roste/Stufen sind 20 mm und hoeher. Danach: Stufe vs. Rost
                        #   vorlaeufig an der Breite - die Rostklasse aus der Namensliste entscheidet danach endgueltig.
                        if d['masse'][2] < 18:
                            d['art'] = 'blech'
                        else:
                            d['art'] = 'gitterroststufe' if d['masse'][1] <= 420 else 'gitterrost'
                        if d['art'] == 'gitterroststufe':
                            # ★ Stufen messen ueber die Box 6 mm zu kurz (Einfassung):
                            #   1194 -> 1200, 1154 -> 1160 (von Paul am echten Modell bestaetigt)
                            d['masse'][0] = round(d['masse'][0] + 6)
                    if d['art'] in ('profil', 'kantprofil') and not d.get('laenge'):
                        mm = masse_aus_obb(m)
                        if mm:
                            d['laenge'] = round(mm[0], 1)   # laengste Kante = Stablaenge
                    if typ == 'IfcMechanicalFastener' and em_bolzen is not None:
                        k = bolzen_index.get(shp.id, -1)
                        if 0 <= k < len(em_bolzen):
                            dm2, ln2, din2, g2 = em_bolzen[k]
                            dm1 = getattr(prod, 'NominalDiameter', None)
                            ln1 = getattr(prod, 'NominalLength', None)
                            if dm1 == dm2 and ln1 == ln2:   # ★ Sicherung: Masse muessen passen
                                if din2: d['din'] = din2
                                if g2 and not d.get('material'): d['material'] = str(g2)
                    if em_baugruppen is not None:
                        z = v.mean(axis=0)
                        dist = np.linalg.norm(em_baugruppen[0] - z, axis=1)
                        k = int(dist.argmin())
                        if dist[k] < 0.005:
                            d['bg'] = em_baugruppen[1][k][0]
                            d['bgnr'] = em_baugruppen[1][k][1]
                    if d['ref'] is None and em_zentren is not None:
                        z = v.mean(axis=0)
                        dist = np.linalg.norm(em_zentren[0] - z, axis=1)
                        k = int(dist.argmin())
                        if dist[k] < 0.005:
                            d['ref'] = em_zentren[1][k]
                    try:
                        vol = float(m.volume)
                        # ★ Roste/Stufen: AS exportiert massive Kloetze - Volumengewicht waere um
                        #   den Faktor 10 zu hoch. Lieber weglassen, bis die Namensliste die
                        #   echten Rostdaten liefert.
                        if vol > 0 and (L not in BETON_LAYER) and d['art'] not in ('schraube', 'anker', 'gitterrost', 'gitterroststufe', 'beton'):
                            d['gewicht'] = round(vol * DICHTE_STAHL, 1)
                    except Exception:
                        pass
                    d['zentrum'] = [round(float(x), 4) for x in ((v.min(axis=0) + v.max(axis=0)) / 2.0)]  # v48: Box-Mitte wie Plugin-Extents
                    teile[kn] = d
                    n += 1
        except Exception:
            fehler += 1
        if not it.next():
            break

    # ★ Nachfass-Durchgang: der Iterator laesst manche Typen still aus (in der EM.11
    #   z.B. alle IfcMechanicalFastener). Jedes Produkt mit Geometrie, das noch fehlt,
    #   wird hier einzeln vermascht - Schweissmuster (IfcFastener) bleiben bewusst weg.
    nachgeholt = 0; nachfehler = 0
    for prod in f.by_type('IfcProduct'):
        typ = prod.is_a()
        if typ in ('IfcOpeningElement', 'IfcFastener', 'IfcSite', 'IfcBuilding',
                   'IfcBuildingStorey', 'IfcElementAssembly', 'IfcGrid'):
            continue
        if not getattr(prod, 'Representation', None):
            continue
        kn = 'E%d' % prod.id()
        if kn in teile: continue
        if ohne_schrauben and typ in SCHRAUBEN_TYPEN: continue
        L = layer_von(prod)
        if ohne_beton and L in BETON_LAYER: continue
        try:
            m3 = None
            if typ == 'IfcMechanicalFastener':
                # ★ EM.11-Schrauben sind nur STRICHSYMBOLE (IfcPolyline) - wir bauen den
                #   Koerper selbst: Symbol-Linie = Achse, NominalDiameter/-Length = Masse.
                m3 = schraube_aus_symbol(f, prod)
            if m3 is None:
                shp3 = ifcopenshell.geom.create_shape(s, prod)
                g3 = shp3.geometry
                v = np.array(g3.verts).reshape(-1, 3)
                fc = np.array(g3.faces).reshape(-1, 3)
                if len(fc) == 0: continue
                m3 = trimesh.Trimesh(vertices=v, faces=fc, process=False)
            v = np.asarray(m3.vertices)
            art0 = art_von(L, typ)
            m3.visual = trimesh.visual.TextureVisuals(material=material_fuer(L, art0))
            szene.add_geometry(m3, node_name=kn, geom_name=kn)
            d = daten_von(prod)
            d['typ'] = typ
            d['layer'] = L
            d['art'] = art0
            if typ == 'IfcMechanicalFastener':
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
                if em_bolzen is not None:
                    k = bolzen_index.get(prod.id(), -1)
                    if 0 <= k < len(em_bolzen):
                        dm2, ln2, din2, g2 = em_bolzen[k]
                        if getattr(prod, 'NominalDiameter', None) == dm2 and getattr(prod, 'NominalLength', None) == ln2:
                            if din2: d['din'] = din2
                            if g2 and not d.get('material'): d['material'] = str(g2)
            if d['art'] in ('blech', 'kantblech'):
                mm = masse_aus_obb(m3)
                if mm: d['masse'] = [round(mm[0]), round(mm[1]), round(mm[2], 1)]
            if d['art'] in ('profil', 'kantprofil') and not d.get('laenge'):
                mm = masse_aus_obb(m3)
                if mm: d['laenge'] = round(mm[0], 1)
            if em_baugruppen is not None:
                z = v.mean(axis=0)
                dist = np.linalg.norm(em_baugruppen[0] - z, axis=1)
                k = int(dist.argmin())
                if dist[k] < 0.005:
                    d['bg'] = em_baugruppen[1][k][0]
                    d['bgnr'] = em_baugruppen[1][k][1]
            if d['ref'] is None and em_zentren is not None:
                z = v.mean(axis=0)
                dist = np.linalg.norm(em_zentren[0] - z, axis=1)
                k = int(dist.argmin())
                if dist[k] < 0.005:
                    d['ref'] = em_zentren[1][k]
            d['zentrum'] = [round(float(x), 4) for x in ((v.min(axis=0) + v.max(axis=0)) / 2.0)]  # v48: Box-Mitte wie Plugin-Extents
            teile[kn] = d
            n += 1; nachgeholt += 1
        except Exception:
            nachfehler += 1
    if nachgeholt or nachfehler:
        print('* Nachfass-Durchgang: %d Teile nachvermascht (%d nicht wandelbar)' % (nachgeholt, nachfehler))
    print('* Vermascht: %d Bauteile | Fehler: %d | %.0fs' % (n, fehler, time.time() - t0))
    if n == 0:
        raise SystemExit('Keine Bauteile in der IFC.')
    glb = basisname + '.glb'
    try:
        szene.export(glb, include_normals=True)  # ★ weiche Normalen: Radien und Rohre rund
    except TypeError:
        szene.export(glb)
    if achsen:
        teile['__achsen__'] = achsen
    return glb, teile

_FLSTAT = {'gesamt': 0, 'leer': 0, 'unplanar': 0}  # v55: misst Pauls 'Flaechen fehlen'

def _flaeche_zerlegen(aussen, loecher):
    _FLSTAT['gesamt'] += 1
    _FLSTAT['letzte_breite'] = 99.0
    try:
        a0 = np.asarray(aussen, dtype=float)
        if len(a0) >= 3:
            n = np.zeros(3)
            for i in range(len(a0)):
                p, q = a0[i], a0[(i + 1) % len(a0)]
                n += np.cross(p, q)
            ln = np.linalg.norm(n)
            if ln > 1e-12:
                P = float(np.sum(np.linalg.norm(np.roll(a0, -1, axis=0) - a0, axis=1)))
                _FLSTAT['letzte_breite'] = ln / max(P, 1e-9)  # v56: 2*Flaeche/Umfang = Streifenbreite
                n2 = n / ln
                d = np.abs((a0 - a0[0]) @ n2)
                diag = float(np.linalg.norm(a0.max(axis=0) - a0.min(axis=0))) or 1.0
                if d.max() > max(1.0, 0.001 * diag):
                    _FLSTAT['unplanar'] += 1
    except Exception:
        pass
    erg = _flaeche_zerlegen_kern(aussen, loecher)
    if erg is None or len(erg) == 0:
        _FLSTAT['leer'] += 1
    return erg

def _flaeche_zerlegen_kern(aussen, loecher):
    """Eine Brep-Flaeche (Aussenkontur + Lochkonturen, Nx3) -> Dreiecke (M,3,3).
    Projektion in die Flaechenebene (Newell-Normale), Earcut mit Loechern."""
    import mapbox_earcut
    ringe = [np.asarray(aussen, dtype=float)] + [np.asarray(h, dtype=float) for h in (loecher or [])]
    ringe = [r for r in ringe if len(r) >= 3]
    if not ringe: return None
    a = ringe[0]
    # Newell-Normale
    n = np.zeros(3)
    for i in range(len(a)):
        p, q = a[i], a[(i + 1) % len(a)]
        n += np.cross(p, q)
    ln = np.linalg.norm(n)
    if ln < 1e-12: return None
    n /= ln
    u = np.cross(n, [0.0, 0.0, 1.0])
    if np.linalg.norm(u) < 1e-6: u = np.cross(n, [0.0, 1.0, 0.0])
    u /= np.linalg.norm(u); v = np.cross(n, u)
    alle = np.vstack(ringe)
    zwei = np.column_stack([alle @ u, alle @ v])
    enden = np.cumsum([len(r) for r in ringe]).astype(np.uint32)
    try:
        tri = mapbox_earcut.triangulate_float64(zwei, enden)
    except Exception:
        return None
    if len(tri) == 0: return None
    return alle[np.asarray(tri, dtype=np.int64)].reshape(-1, 3, 3)

def _wandle_geo(geo_pfad, json_pfad, ohne_schrauben=False):
    """★ DIREKT-Rohformat: T E<id> / L x y z ... (Aussenkontur) / H x y z ... (Loch,
    gehoert zur letzten L-Zeile). Millimeter. Zerlegung hier in der Cloud - das
    Plugin bleibt dumm und robust."""
    import json as _js, time as _t
    t0 = _t.time()
    meta = {}
    if json_pfad and os.path.exists(json_pfad):
        try:
            with open(json_pfad, encoding='utf-8', errors='replace') as fh:
                meta = _js.load(fh)
        except Exception as ex:
            print('! direkt.json unlesbar (%s) - baue ohne Steckbriefe weiter.' % ex)
    skal = 0.001 if str(meta.get('einheit', 'mm')).lower().startswith('mm') else 1.0
    info = meta.get('teile', {}) or {}
    try:
        KONFIG.update({k: v for k, v in (meta.get('konfig') or {}).items() if v})
        if KONFIG: print('* Dialog-Konfig uebernommen: %s' % ', '.join(sorted(KONFIG)))
    except Exception:
        pass

    ART_ERSATZ = {'profil': 'AS_Tr?ger', 'kantprofil': 'AS_Kanttr?ger', 'blech': 'AS_Bleche',
                  'schraube': 'AS_Schrauben', 'anker': 'AS_Ankerk?fige', 'kopfbolzen': 'AS_Kopfbolzen',
                  'gitterrost': 'AS_Gitterroste', 'gitterroststufe': 'AS_Stufen',
                  'sonderteil': 'AS_Sonderteile', 'beton': 'AS_Betondecken'}
    KLASSE_ART = [('foldedbeam', 'kantprofil'), ('foldedplate', 'kantblech'), ('bentbeam', 'kantprofil'),
                  ('polybeam', 'profil'), ('compoundbeam', 'profil'), ('unfolded', 'kantblech'),
                  ('grating', 'gitterrost'), ('specialpart', 'sonderteil'), ('anchor', 'anker'),
                  ('shearstud', 'kopfbolzen'), ('connector', 'kopfbolzen'), ('bolt', 'schraube'),
                  ('screw', 'schraube'), ('plate', 'blech'), ('beam', 'profil'),
                  ('wall', 'beton'), ('slab', 'beton'), ('concrete', 'beton')]
    def klasse_art(k):
        k = (k or '').lower()
        for schl, art in KLASSE_ART:
            if schl in k: return art
        return None
    material_cache = {}
    def material_fuer(layer, art, eigen=None):
        # ★ Objekt-Farbe (nicht 'Von Layer') schlaegt die Layertabelle - fuer alle Bauteile
        if layer not in LAYER_FARBE and art in ART_ERSATZ:
            layer = ART_ERSATZ[art]
        schl = (layer, eigen)
        if schl in material_cache: return material_cache[schl]
        col = LAYER_FARBE.get(norm_layer(layer), STANDARD_FARBE)
        zusatz = ''
        if eigen:
            try:
                h = str(eigen).lstrip('#')
                col = (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))
                zusatz = '_F' + h.upper()
            except Exception:
                zusatz = ''
        mat = trimesh.visual.material.PBRMaterial(
            baseColorFactor=[col[0]/255.0, col[1]/255.0, col[2]/255.0, 1.0],
            metallicFactor=0.15, roughnessFactor=0.7)
        mat.name = 'GOKOBA_ACI_%d_%s%s' % (LAYER_ACI.get(norm_layer(layer), 0), re.sub(r'[^A-Za-z0-9_]', '_', str(layer)), zusatz)
        material_cache[schl] = mat
        return mat

    szene = trimesh.Scene(); teile = {}; n = 0; fehler = 0
    kn = None; dreiecke = []; aussen = None; loecher = []; fl_ntris = []; fl_breitL = []; fl_hatLoch = []; fl_lochB = []

    def _fl_ab():
        nonlocal flLeer
        if aussen is not None:
            t3 = _flaeche_zerlegen(aussen, loecher)
            if t3 is not None:
                dreiecke.append(t3)
                fl_ntris.append(len(t3))
                fl_breitL.append(_FLSTAT.get('letzte_breite', 99.0) > 12.0)  # v56: >12mm = breite Ebene
                fl_hatLoch.append(bool(loecher))  # v58: Deckel-Erkennung
                fl_lochB.append([(np.asarray(lo, dtype=float).min(axis=0), np.asarray(lo, dtype=float).max(axis=0)) for lo in (loecher or [])])  # v65: Lochring-Boxen
            else: flLeer += 1

    def _teil_ab():
        nonlocal n, fehler
        if kn is None: return
        try:
            if not dreiecke: return
            t3 = np.vstack(dreiecke) * skal
            va = t3.reshape(-1, 3)
            fa = np.arange(len(va), dtype=np.int64).reshape(-1, 3)
            m = trimesh.Trimesh(vertices=va, faces=fa, process=False)
            m.merge_vertices()
            fl_id = np.repeat(np.arange(len(fl_ntris)), fl_ntris) if fl_ntris else None
            fl_breit = np.array(fl_breitL, dtype=bool) if fl_breitL else None
            try:  # v56: exakte Doppel-Dreiecke raus (Flimmer-Fragmente)
                srt = np.sort(m.faces, axis=1)
                _, uidx = np.unique(srt, axis=0, return_index=True)
                if len(uidx) < len(m.faces):
                    maske = np.zeros(len(m.faces), dtype=bool); maske[uidx] = True
                    _FLSTAT['doppel'] = _FLSTAT.get('doppel', 0) + int(len(m.faces) - len(uidx))
                    m.update_faces(maske)
                    if fl_id is not None: fl_id = fl_id[maske]
            except Exception:
                pass
            try:  # v56: Orientierung VOR den Knick-Normalen richten (vorher andersrum!)
                m.fix_normals()
                if m.is_watertight and m.volume < 0: m.invert()
                if m.is_watertight: _FLSTAT['dicht'] = _FLSTAT.get('dicht', 0) + 1
            except Exception:
                pass
            try:  # v57: koplanare Doppel-Flaechen je Teil zaehlen (Messung Rest-Flimmern)
                if fl_id is not None and len(m.faces) == len(fl_id):
                    fnm = m.face_normals; ctr = m.triangles_center
                    ebenen = {}
                    for fid in np.unique(fl_id):
                        mk = fl_id == fid
                        nn = fnm[mk].mean(axis=0); ln2 = float(np.linalg.norm(nn))
                        if ln2 < 1e-9: continue
                        nn = nn / ln2
                        if nn[int(np.argmax(np.abs(nn)))] < 0: nn = -nn
                        dd = float(np.dot(nn, ctr[mk].mean(axis=0)))
                        key = (round(float(nn[0]), 2), round(float(nn[1]), 2), round(float(nn[2]), 2), round(dd, 3))  # v61: 1mm-Toleranz
                        ebenen.setdefault(key, []).append(int(fid))
                    _FLSTAT['koplanar'] = _FLSTAT.get('koplanar', 0) + sum(len(v) - 1 for v in ebenen.values() if len(v) > 1)
                    # v58: Stanz-Deckel raus - koplanare, LOCHLOSE, deutlich kleinere Flaechen,
                    #   die raeumlich in der groessten Flaeche derselben Ebene liegen.
                    weg = np.zeros(len(m.faces), dtype=bool)
                    A3 = m.area_faces
                    hatL = np.array(fl_hatLoch, dtype=bool) if fl_hatLoch else None
                    for key, fids in ebenen.items():
                        if len(fids) < 2: continue
                        fl_area = {fid: float(A3[fl_id == fid].sum()) for fid in fids}
                        def _mitLoch(f):
                            return bool(hatL is not None and f < len(hatL) and hatL[f])
                        lochIds = [f for f in fids if _mitLoch(f)]
                        # v64: LOCH-VORRANG - Referenz ist die groesste GELOCHTE Flaeche der Ebene
                        #   (die lochlose Kopie darueber ist die groessere und uebermalte sonst das Loch)
                        ref = max(lochIds, key=lambda f: fl_area[f]) if lochIds else max(fids, key=lambda f: fl_area[f])
                        mg = fl_id == ref
                        gmin = m.triangles.reshape(-1, 3)[np.repeat(mg, 3)].min(axis=0) - 0.002
                        gmax = m.triangles.reshape(-1, 3)[np.repeat(mg, 3)].max(axis=0) + 0.002
                        for fid in fids:
                            if fid == ref: continue
                            if _mitLoch(fid): continue  # gelochte Flaechen nie verwerfen
                            mk2 = fl_id == fid
                            pts2 = m.triangles.reshape(-1, 3)[np.repeat(mk2, 3)]
                            innen = bool(np.all(pts2 >= gmin) and np.all(pts2 <= gmax))
                            if not innen: continue
                            if fl_area[fid] <= 1.35 * fl_area[ref]:  # Deckel (klein) UND Zwilling/Uebermaler (~gleich/groesser)
                                weg |= mk2
                                _FLSTAT['deckel'] = _FLSTAT.get('deckel', 0) + 1
                    if weg.any():
                        m.update_faces(~weg)
                        fl_id = fl_id[~weg]
            except Exception:
                pass
            try:  # v65: Stanz-Deckel IN DER LOCHTIEFE - Umriss ~ Lochring => weg (Achse frei)
                alleLB = []
                for li in fl_lochB:
                    for lmn, lmx in li:
                        alleLB.append((lmn * skal, lmx * skal))
                if alleLB and fl_id is not None and len(m.faces) == len(fl_id):
                    A5 = m.area_faces
                    tri5 = m.triangles.reshape(-1, 3)
                    weg2 = np.zeros(len(m.faces), dtype=bool)
                    for fid in np.unique(fl_id):
                        mk5 = fl_id == fid
                        if float(A5[mk5].sum()) > 0.02:
                            continue  # v68: hatL-Sperre entfernt (Halbmond-Deckel sind Ring-Stuecke MIT Loch);
                                      #   der Groessen-Match schuetzt die echte Traegerflaeche
                        p5 = tri5[np.repeat(mk5, 3)]
                        fmin = p5.min(axis=0); fmax = p5.max(axis=0)
                        fc = (fmin + fmax) / 2.0; fs = fmax - fmin
                        for lmn, lmx in alleLB:
                            lc = (lmn + lmx) / 2.0; ls = lmx - lmn
                            ax = int(np.argmin(ls))
                            tol = ls * 0.5 + fs * 0.5 + 0.002
                            tol[ax] += 0.012
                            if not np.all(np.abs(fc - lc) <= tol):
                                continue
                            gut = True
                            for q in range(3):
                                if q == ax:
                                    continue
                                if abs(float(fs[q] - ls[q])) > max(0.5 * float(ls[q]), 0.004):
                                    gut = False; break
                            if gut:
                                weg2 |= mk5
                                _FLSTAT['lochdeckel'] = _FLSTAT.get('lochdeckel', 0) + 1
                                break
                    if weg2.any():
                        m.update_faces(~weg2)
                        fl_id = fl_id[~weg2]
            except Exception:
                pass
            m = _knick_normalen(m, fl_id=fl_id, fl_breit=fl_breit)
            d0 = info.get(kn, {}) or {}
            L = d0.get('layer')
            art = ART_LAYER.get(norm_layer(L)) or klasse_art(d0.get('klasse')) or 'sonstiges'
            if ohne_schrauben and art in ('schraube', 'anker', 'kopfbolzen'): return
            m.visual = trimesh.visual.TextureVisuals(material=material_fuer(L, art, d0.get('farbe')))
            szene.add_geometry(m, node_name=kn, geom_name=kn)
            d = {'ref': _pseudo(d0.get('pos')), 'profil': saniere_profil(d0.get('profil')), 'farbe': d0.get('farbe'), 'familie': d0.get('familie'),
                 'material': ('Alu' if str(d0.get('material') or '').strip().lower() == 'al' else d0.get('material')), 'laenge': d0.get('laenge'),
                 'gewicht': d0.get('gewicht'), 'typ': d0.get('klasse'), 'layer': L,
                 'art': art, 'roh': d0.get('name')}
            if d0.get('name'): d['name'] = str(d0['name'])
            a0 = d0.get('attrs')
            if a0: d['attrs'] = [str(x) for x in a0]
            if d0.get('bgnr'): d['bg'] = str(d0['bgnr']); d['bgnr'] = str(d0['bgnr'])
            if d0.get('bgname'): d['bgname'] = str(d0['bgname'])
            for feld in ('din', 'groesse', 'hersteller', 'masche', 'tragstab'):
                if d0.get(feld): d[feld] = str(d0[feld])
            if d0.get('bestand'): d['bestand'] = True
            if d0.get('beschichtung'): d['beschichtung'] = d0.get('beschichtung')  # v49
            if d0.get('attrs') and not d.get('attrs'):
                d['attrs'] = [w if w else '' for w in d0.get('attrs')]  # v50: Attribute direkt aus der json
            if d0.get('blockname') and not d.get('name'):
                name_deute(d, d0.get('blockname'))  # v50: Sonderteil-Blockname direkt aus der json
            if art in ('blech', 'kantblech', 'gitterrost', 'gitterroststufe'):
                # ★ Masse aus dem AS-Dialog haben Vorrang - die orientierte Box irrt
                #   bei Kantblechen und angeschweissten Anbauteilen (falsche 'Dicke').
                gl = [d0.get('blechlaenge'), d0.get('blechbreite'), d0.get('dicke')]
                if all(isinstance(x, (int, float)) and x > 0 for x in gl):
                    d['masse'] = [round(gl[0]), round(gl[1]), round(gl[2], 2)]  # v49: Dicke exakt (8.76 blieb sonst nicht 8.76)
                else:
                    mm = masse_aus_obb(m)
                    di = d0.get('dicke')
                    if mm:
                        d['masse'] = [round(mm[0]), round(mm[1]),
                                      round(di if isinstance(di, (int, float)) and di > 0 else mm[2], 2)]
            if art in ('profil', 'kantprofil') and not d.get('laenge'):
                mm = masse_aus_obb(m)
                if mm: d['laenge'] = round(mm[0], 1)
            try:  # v66: Netto-Projektionsflaeche (Ausschnitte abgezogen) fuer Roste/Werkstoffe
                ext6 = m.bounds[1] - m.bounds[0]
                ax6 = int(np.argmin(ext6))
                mk6 = np.abs(m.face_normals[:, ax6]) > 0.9
                fl6 = float(m.area_faces[mk6].sum()) / 2.0
                if fl6 > 0:
                    d['flaeche'] = round(fl6, 3)
            except Exception:
                pass
            if art in ('anker', 'kopfbolzen'):
                d['gewicht'] = None  # v62: Anker ohne Gewicht (Pauls Vorgabe)
            if d.get('gewicht') is None and art not in ('anker', 'kopfbolzen'):
                _bwsT = ((d.get('material') or '') + ' ' + (str(L or ''))).lower()
                if 'bws' in _bwsT or 'werkstein' in _bwsT:
                    gB = d0.get('gewicht_as')
                    if isinstance(gB, (int, float)) and gB > 0:
                        d['gewicht'] = round(gB / 1000.0, 2)  # v68: AS-Weight in GRAMM; 400x400x40 -> 16,00 kg
                    else:
                        vB = d0.get('volumen')
                        if isinstance(vB, (int, float)) and vB > 0:
                            d['gewicht'] = round(vB * 1e-9 * 2500.0, 1)  # Rueckfall: Pauls AS-Dichte 2,5 kg/dm3
                elif 'glas' in _bwsT or 'mineralit' in _bwsT:
                    gB = d0.get('gewicht_as')
                    if isinstance(gB, (int, float)) and gB > 0:
                        d['gewicht'] = round(gB / 1000.0, 2)  # v68: AS-Weight in GRAMM
                    else:
                        vB = d0.get('volumen')
                        if isinstance(vB, (int, float)) and vB > 0:
                            dichte6 = 2450.0 if 'mineralit' in _bwsT else 2500.0  # Mineralit-Infoblatt 2,45 g/cm3; Glas 2,5
                            d['gewicht'] = round(vB * 1e-9 * dichte6, 2)
            if d.get('gewicht') is None and art in ('profil', 'kantprofil', 'blech', 'kantblech', 'sonderteil'):
                gAS = d0.get('gewicht_as')  # v61: AS-eigenes Gewicht = exakt wie Pauls AS-Liste
                if isinstance(gAS, (int, float)) and gAS > 0:
                    d['gewicht'] = round(gAS / 1000.0, 2)  # v68: AS-Weight in GRAMM (gemessen: BWS-Referenz 16000)
            if d.get('gewicht') is None and art in ('profil', 'kantprofil', 'blech', 'kantblech', 'sonderteil'):
                vAS = d0.get('volumen')  # v60: exaktes AS-Volumen (mm3) - unabhaengig von Wasserdichtheit
                if isinstance(vAS, (int, float)) and vAS > 0:
                    mAS = (d.get('material') or '').strip().lower()
                    dAS = 2700.0 if mAS in ('al', 'alu', 'aluminium') or mAS.startswith('almg') or mAS.startswith('en aw') else DICHTE_STAHL
                    d['gewicht'] = round(vAS * 1e-9 * dAS, 1)
            if d.get('gewicht') is None and art in ('profil', 'kantprofil', 'blech', 'kantblech', 'sonderteil'):
                try:
                    if m.is_watertight:
                        vol = float(abs(m.volume))
                        mat_u = (d.get('material') or '').strip().lower()
                        dichte = 2700.0 if mat_u in ('al', 'alu', 'aluminium') or mat_u.startswith('almg') or mat_u.startswith('en aw') else DICHTE_STAHL
                        if vol > 0: d['gewicht'] = round(vol * dichte, 1)
                except Exception:
                    pass
            d['zentrum'] = [round(float(x), 4) for x in ((va.min(axis=0) + va.max(axis=0)) / 2.0)]  # v48: Box-Mitte wie Plugin-Extents
            teile[kn] = d
            n += 1
        except Exception:
            fehler += 1

    def _tripel(z):
        w = z.split()[1:]
        k = np.asarray([float(x) for x in w], dtype=float)
        return k.reshape(-1, 3)

    kaputt = 0
    nT = nL = nH = 0; flLeer = 0; probeZeilen = []
    with open(geo_pfad, encoding='utf-8', errors='replace') as fh:
        for zeile in fh:
            try:
                z = zeile.strip()
                if not z: continue
                if z[0] == 'T':
                    nT += 1
                    _fl_ab(); _teil_ab()
                    kn = z.split()[1] if len(z.split()) > 1 else None
                    dreiecke = []; aussen = None; loecher = []; fl_ntris = []; fl_breitL = []; fl_hatLoch = []; fl_lochB = []
                elif z[0] == 'L':
                    nL += 1
                    if len(probeZeilen) < 3: probeZeilen.append(z[:140])
                    _fl_ab()
                    aussen = _tripel(z); loecher = []
                elif z[0] == 'H':
                    nH += 1
                    if aussen is not None: loecher.append(_tripel(z))
                elif len(probeZeilen) < 3:
                    probeZeilen.append('UNBEKANNT: ' + z[:140])
            except Exception:
                kaputt += 1
    if kaputt:
        print('! %d unlesbare Geo-Zeilen uebersprungen.' % kaputt)
    _fl_ab(); _teil_ab()

    print('* DIREKT (geo) vermascht: %d Bauteile | Fehler: %d | %.0fs' % (n, fehler, _t.time() - t0))
    print('* DIREKT-Diagnose: T=%d L=%d H=%d | Flaechen ohne Zerlegung: %d | unlesbare Zeilen: %d' % (nT, nL, nH, flLeer, kaputt))
    for pz in probeZeilen:
        print('* Probezeile: %s' % pz)
    if n == 0:
        raise SystemExit('Direkt-Geo-Paket enthaelt keine Bauteile - siehe Diagnosezeilen daruber.')
    achsen = []
    for a in (meta.get('achsen') or []):
        try:
            achsen.append({'tag': str(a.get('tag', '?')), 'p': [float(x) * skal for x in (a.get('p') or [])][:6]})
        except Exception:
            pass
    glb = os.path.splitext(geo_pfad)[0] + '.glb'
    try:
        szene.export(glb, include_normals=True)  # ★ weiche Normalen: Radien und Rohre rund
    except TypeError:
        szene.export(glb)
    if achsen:
        teile['__achsen__'] = achsen
    return glb, teile

def wandle_direkt(obj_pfad, json_pfad, ohne_schrauben=False):
    """★ DIREKT-EXPORTER-Paket: direkt.obj (Gruppen 'o E<handle>', Millimeter) +
    direkt.json (Steckbrief je Teil). Das Plugin vernetzt selbst - hier wird nur
    verpackt und mit denselben Kaertchen-Daten versorgt wie auf dem IFC-Weg."""
    import json as _js, time as _t
    t0 = _t.time()
    meta = {}
    if json_pfad and os.path.exists(json_pfad):
        with open(json_pfad, encoding='utf-8') as fh:
            meta = _js.load(fh)
    skal = 0.001 if str(meta.get('einheit', 'mm')).lower().startswith('mm') else 1.0
    info = meta.get('teile', {}) or {}
    try:
        KONFIG.update({k: v for k, v in (meta.get('konfig') or {}).items() if v})
        if KONFIG: print('* Dialog-Konfig uebernommen: %s' % ', '.join(sorted(KONFIG)))
    except Exception:
        pass
    geo_pfad = os.path.join(os.path.dirname(obj_pfad), 'direkt.geo')
    if os.path.exists(geo_pfad):
        return _wandle_geo(geo_pfad, json_pfad, ohne_schrauben)
    # ★ Eigener Mini-Parser statt trimesh.load: unser Format (o/v/f), damit die
    #   Objekt-Trennung nie von Bibliotheksversionen abhaengt.
    paare = []
    _vs = []; _fs = []; _name = None
    def _abschluss():
        if _name is not None and _fs:
            va = np.asarray(_vs, dtype=float)
            fa = np.asarray(_fs, dtype=np.int64) - 1
            # ★ nur die Punkte dieses Teils mitnehmen (OBJ-Indizes sind global)
            uq, inv = np.unique(fa.reshape(-1), return_inverse=True)
            paare.append((_name, trimesh.Trimesh(vertices=va[uq],
                                                 faces=inv.reshape(-1, 3), process=False)))
    with open(obj_pfad, encoding='utf-8', errors='replace') as fh:
        for zeile in fh:
            z = zeile.strip()
            if z.startswith('o ') or z.startswith('g '):
                _abschluss(); _name = z[2:].strip(); _fs = []
            elif z.startswith('v '):
                p = z.split()
                _vs.append((float(p[1]), float(p[2]), float(p[3])))
            elif z.startswith('f '):
                ix = [int(w.split('/')[0]) for w in z.split()[1:]]
                for k in range(1, len(ix) - 1):
                    _fs.append((ix[0], ix[k], ix[k + 1]))
    _abschluss()
    if not paare and _fs:
        _name = 'E0'; _abschluss()

    ART_ERSATZ = {'profil': 'AS_Tr?ger', 'kantprofil': 'AS_Kanttr?ger', 'blech': 'AS_Bleche',
                  'schraube': 'AS_Schrauben', 'anker': 'AS_Ankerk?fige', 'kopfbolzen': 'AS_Kopfbolzen',
                  'gitterrost': 'AS_Gitterroste', 'gitterroststufe': 'AS_Stufen',
                  'sonderteil': 'AS_Sonderteile', 'beton': 'AS_Betondecken'}
    KLASSE_ART = [('foldedbeam', 'kantprofil'), ('foldedplate', 'kantblech'), ('bentbeam', 'kantprofil'),
                  ('grating', 'gitterrost'), ('specialpart', 'sonderteil'), ('anchor', 'anker'),
                  ('shearstud', 'kopfbolzen'), ('connector', 'kopfbolzen'), ('bolt', 'schraube'),
                  ('screw', 'schraube'), ('plate', 'blech'), ('beam', 'profil'),
                  ('wall', 'beton'), ('slab', 'beton')]
    def klasse_art(k):
        k = (k or '').lower()
        for schl, art in KLASSE_ART:
            if schl in k: return art
        return None
    material_cache = {}
    def material_fuer(layer, art, eigen=None):
        # ★ Objekt-Farbe (nicht 'Von Layer') schlaegt die Layertabelle - fuer alle Bauteile
        if layer not in LAYER_FARBE and art in ART_ERSATZ:
            layer = ART_ERSATZ[art]
        schl = (layer, eigen)
        if schl in material_cache: return material_cache[schl]
        col = LAYER_FARBE.get(norm_layer(layer), STANDARD_FARBE)
        zusatz = ''
        if eigen:
            try:
                h = str(eigen).lstrip('#')
                col = (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))
                zusatz = '_F' + h.upper()
            except Exception:
                zusatz = ''
        mat = trimesh.visual.material.PBRMaterial(
            baseColorFactor=[col[0]/255.0, col[1]/255.0, col[2]/255.0, 1.0],
            metallicFactor=0.15, roughnessFactor=0.7)
        mat.name = 'GOKOBA_ACI_%d_%s%s' % (LAYER_ACI.get(norm_layer(layer), 0), re.sub(r'[^A-Za-z0-9_]', '_', str(layer)), zusatz)
        material_cache[schl] = mat
        return mat

    szene = trimesh.Scene(); teile = {}; n = 0; fehler = 0
    for kn, m in paare:
        try:
            if not str(kn).startswith('E'):
                kn = 'E%s' % kn
            v = np.asarray(m.vertices, dtype=float) * skal
            m = trimesh.Trimesh(vertices=v, faces=np.asarray(m.faces), process=False)
            d0 = info.get(kn, {}) or {}
            L = d0.get('layer')
            art = ART_LAYER.get(norm_layer(L)) or klasse_art(d0.get('klasse')) or 'sonstiges'
            if ohne_schrauben and art in ('schraube', 'anker', 'kopfbolzen'):
                continue
            m.visual = trimesh.visual.TextureVisuals(material=material_fuer(L, art, d0.get('farbe')))
            szene.add_geometry(m, node_name=kn, geom_name=kn)
            d = {'ref': _pseudo(d0.get('pos')), 'profil': saniere_profil(d0.get('profil')), 'farbe': d0.get('farbe'), 'familie': d0.get('familie'),
                 'material': ('Alu' if str(d0.get('material') or '').strip().lower() == 'al' else d0.get('material')), 'laenge': d0.get('laenge'),
                 'gewicht': d0.get('gewicht'), 'typ': d0.get('klasse'), 'layer': L,
                 'art': art, 'roh': d0.get('name')}
            if d0.get('name'): d['name'] = str(d0['name'])
            a = d0.get('attrs')
            if a: d['attrs'] = [str(x) for x in a]
            if d0.get('bgnr'):
                d['bg'] = str(d0['bgnr']); d['bgnr'] = str(d0['bgnr'])
            if d0.get('bgname'): d['bgname'] = str(d0['bgname'])
            for feld in ('din', 'groesse', 'hersteller', 'masche', 'tragstab'):
                if d0.get(feld): d[feld] = str(d0[feld])
            if d0.get('bestand'): d['bestand'] = True
            if d0.get('beschichtung'): d['beschichtung'] = d0.get('beschichtung')  # v49
            if d0.get('attrs') and not d.get('attrs'):
                d['attrs'] = [w if w else '' for w in d0.get('attrs')]  # v50: Attribute direkt aus der json
            if d0.get('blockname') and not d.get('name'):
                name_deute(d, d0.get('blockname'))  # v50: Sonderteil-Blockname direkt aus der json
            if art in ('blech', 'kantblech', 'gitterrost', 'gitterroststufe'):
                # ★ Masse aus dem AS-Dialog haben Vorrang - die orientierte Box irrt
                #   bei Kantblechen und angeschweissten Anbauteilen (falsche 'Dicke').
                gl = [d0.get('blechlaenge'), d0.get('blechbreite'), d0.get('dicke')]
                if all(isinstance(x, (int, float)) and x > 0 for x in gl):
                    d['masse'] = [round(gl[0]), round(gl[1]), round(gl[2], 2)]  # v49: Dicke exakt (8.76 blieb sonst nicht 8.76)
                else:
                    mm = masse_aus_obb(m)
                    di = d0.get('dicke')
                    if mm:
                        d['masse'] = [round(mm[0]), round(mm[1]),
                                      round(di if isinstance(di, (int, float)) and di > 0 else mm[2], 2)]
            if art in ('profil', 'kantprofil') and not d.get('laenge'):
                mm = masse_aus_obb(m)
                if mm: d['laenge'] = round(mm[0], 1)
            if d.get('gewicht') is None and art in ('profil', 'kantprofil', 'blech', 'kantblech'):
                try:
                    if m.is_watertight:
                        vol = float(m.volume)
                        if vol > 0: d['gewicht'] = round(vol * DICHTE_STAHL, 1)
                except Exception:
                    pass
            d['zentrum'] = [round(float(x), 4) for x in ((v.min(axis=0) + v.max(axis=0)) / 2.0)]  # v48: Box-Mitte wie Plugin-Extents
            teile[kn] = d
            n += 1
        except Exception:
            fehler += 1
    print('* DIREKT vermascht: %d Bauteile | Fehler: %d | %.0fs' % (n, fehler, _t.time() - t0))
    if n == 0:
        raise SystemExit('Direkt-Paket enthaelt keine Bauteile.')
    achsen = []
    for a in (meta.get('achsen') or []):
        try:
            achsen.append({'tag': str(a.get('tag', '?')), 'p': [float(x) * skal for x in (a.get('p') or [])][:6]})
        except Exception:
            pass
    glb = os.path.splitext(obj_pfad)[0] + '.glb'
    try:
        szene.export(glb, include_normals=True)  # ★ weiche Normalen: Radien und Rohre rund
    except TypeError:
        szene.export(glb)
    if achsen:
        teile['__achsen__'] = achsen
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

    # ★ DIREKT-EXPORTER-Paket hat Vorrang: eigene Geometrie, null AS-Exportbefehle.
    direkt_glb = None
    dobj = os.path.join(args.input_dir, 'direkt.obj')
    if os.path.exists(os.path.join(args.input_dir, 'direkt.geo')) and not os.path.exists(dobj):
        dobj = os.path.join(args.input_dir, 'direkt.geo').replace('direkt.geo', 'direkt.obj')  # Pfadbasis fuer wandle_direkt
    if os.path.exists(dobj) or os.path.exists(os.path.join(args.input_dir, 'direkt.geo')):
        print('* DIREKT: eigenes Exportpaket erkannt - Geometrie kommt vom Plugin selbst.')
        direkt_glb = wandle_direkt(dobj, os.path.join(args.input_dir, 'direkt.json'), args.ohne_schrauben)
    ifcs = sorted(glob.glob(os.path.join(args.input_dir, '*.ifc')))
    haupt = [p for p in ifcs if not ist_em11(p)]
    em11 = [p for p in ifcs if ist_em11(p)]
    if direkt_glb is None and not haupt and em11:
        # ★ EM.11-Notweg: der AS-IFC2x3-Export haengt an manchen Modellen (Kanttraeger).
        #   Dann baut der Viewer die Geometrie direkt aus der EM.11-Datei. Bauteilarten
        #   kommen ueber den Entity-Rueckfall in art_von(); der Positionsabgleich laeuft
        #   gegen dieselbe Datei und trifft damit jedes Teil.
        print('* NOTWEG: keine IFC2x3-Datei - Viewer wird komplett aus der EM.11 gebaut.')
        haupt = em11
    if not haupt and direkt_glb is None:
        raise SystemExit('Keine IFC-Datei im Ordner gefunden.')
    ifc = haupt[0] if haupt else None
    em = em11[0] if em11 else None
    if not em:
        print('Hinweis: keine EM.11-Datei dabei - Gelaenderteile bekommen keine Positionsnummern.')

    namen_pos, namen_ort, namen_bg = lese_namen(args.input_dir)
    if namen_pos or namen_ort or namen_bg:
        print('* Namensliste: %d Positionen, %d mit Koordinate, %d Baugruppen-Bezeichnungen'
              % (len(namen_pos), len(namen_ort), len(namen_bg)))
    if direkt_glb is not None:
        glb, teile = direkt_glb
    else:
        glb, teile = wandle(ifc, em, args.ohne_schrauben, args.ohne_beton)
    # ── Namensliste verheiraten: erst Position, dann Schwerpunkt (Sonderteile) ──
    if namen_pos or namen_ort:
        ort_pkt = np.array([o[0] for o in namen_ort]) if namen_ort else None
        getroffen = 0
        fehl_dist = []  # v49: Diagnose fuer Orts-Abgleich
        for kn, d in teile.items():
            if not isinstance(d, dict): continue
            e = None
            if d.get('ref') and d['ref'] in namen_pos:
                e = namen_pos[d['ref']]
            elif ort_pkt is not None and d.get('zentrum') is not None:
                dist = np.linalg.norm(ort_pkt - np.array(d['zentrum']), axis=1)
                k = int(dist.argmin())
                if dist[k] < 0.01:
                    e = namen_ort[k][1]
                else:
                    fehl_dist.append(round(float(dist[k]) * 1000.0, 1))  # v49: naechste Distanz in mm
            if e:
                if e.get('klasse') != 'Attr':
                    name_deute(d, e['name']); getroffen += 1
                a = [w for w in (e.get('attrs') or [])]
                if any(a):
                    d['attrs'] = [w for w in a if w] and a  # volle Liste mit Leerstellen fuer Index-Treue
                    while d['attrs'] and not d['attrs'][-1]: d['attrs'].pop()
            if namen_bg and d.get('bgnr') and d['bgnr'] in namen_bg:
                d['bgname'] = namen_bg[d['bgnr']]
        print('* Namensliste zugeordnet: %d Bauteile (v49, Box-Mitte + Fehl-Distanz-Diagnose)' % getroffen)
        if fehl_dist:
            print('* Orts-Abgleich OHNE Treffer: %d Teile, naechste Distanz [mm]: %s' % (len(fehl_dist), sorted(fehl_dist)[:8]))
    # ★ Rost oder Stufe: die Rostklasse aus dem Modell entscheidet, nicht die Breite.
    #   Prioritaet: Klasse sagt Grating/Graiting -> Rost; Klasse/Beschreibung sagt Stufe -> Stufe;
    #   sonst Pauls Standard-Stufentiefen 240/270/305; sonst bleibt die bisherige Zuordnung.
    for d in teile.values():  # v68: ALLE Teile (der Stufen-Loop darunter filtert auf Rost/Stufe)
        if not isinstance(d, dict):
            continue
        typ8 = ((d.get('typ') or '')).lower()
        if d.get('art') in ('gitterrost', 'gitterroststufe') and 'grating' not in typ8 and 'graiting' not in typ8:
            # Text-'Stufen' OHNE Grating-Klasse = angeschweisste Laschen o.ae. -> zurueckstufen
            d['art'] = 'blech' if 'plate' in typ8 else ('profil' if 'beam' in typ8 else d['art'])
        if d.get('art') in ('blech', 'kantblech') and 'stufe' in ((d.get('bgname') or '')).lower():
            d['gewicht'] = None  # an Gitterroststufen angeschweisste Laschen = Teil des Zukaufteils
        _ns = ((d.get('material') or '') + ' ' + (d.get('layer') or '') + ' ' + (d.get('roh') or '')).lower()
        if (d.get('art') == 'sonderteil'
                or any(w in _ns for w in ('glas', 'thermostop', 'bws', 'werkstein', 'mineralit', 'trespa',
                                          'gummi', 'plattenlager', 'abdeckleist', 'dachbelag'))
                or (('alu' in _ns) and ('wann' in _ns or 'flach' in _ns))):
            d['nichtstahl'] = 1  # zaehlt NICHT in die Profile+Bleche-Summe (Kaertchen-Gewicht bleibt)
        try:  # Zerlegung der Gewichtssumme fuer bericht.txt
            _g6 = d.get('gewicht') or 0
            _l6 = ((d.get('layer') or '')).lower()
            _gel6 = ('gelaender' in _l6) or ('geländer' in _l6) or ('gelander' in _l6)
            if d.get('nichtstahl'):
                _FLSTAT['gw_nichtstahl'] = _FLSTAT.get('gw_nichtstahl', 0.0) + _g6
            elif d.get('art') in ('profil', 'kantprofil'):
                _FLSTAT['gw_prof_gel' if _gel6 else 'gw_prof'] = _FLSTAT.get('gw_prof_gel' if _gel6 else 'gw_prof', 0.0) + _g6
            elif d.get('art') in ('blech', 'kantblech'):
                _FLSTAT['gw_blech_gel' if _gel6 else 'gw_blech'] = _FLSTAT.get('gw_blech_gel' if _gel6 else 'gw_blech', 0.0) + _g6
        except Exception:
            pass
    umsortiert = 0
    for d in teile.values():
        if not isinstance(d, dict) or d.get('art') not in ('gitterrost', 'gitterroststufe'):
            continue
        # ★ NUR der eigene Rohname zaehlt: Baugruppenname/Attribute wuerden das
        #   'Stufe'-Wort auf angeschweisste Laschen vererben (305x70-Bleche als Stufen!).
        quelle = (d.get('roh') or '').lower()
        a1t = ''
        try:
            a1t = ((d.get('attrs') or [None])[0] or '').lower()
        except Exception:
            a1t = ''
        qt = quelle + ' ' + a1t  # v62: Rohname + Benutzerattribut 1 des Teils SELBST (nie bgname - Laschen-Falle!)
        typ_l = ((d.get('typ') or '')).lower()
        alt_art = d['art']
        m = d.get('masse') or []
        tiefe = m[1] if len(m) > 1 else 0
        if 'grating' in typ_l or 'graiting' in typ_l:
            # v64: NUR echte Grating-Objekte werden Rost/Stufe - die 311x70-Anschraub-
            #   laschen (Plate) liefen sonst ueber ihren Attributtext als Stufen mit.
            if 'stufe' in qt or 'step' in qt or 'tread' in qt:
                d['art'] = 'gitterroststufe'
            elif tiefe and any(abs(tiefe - st) <= 3 for st in (240, 270, 305)) and (m[0] if m else 0) <= 1700:
                d['art'] = 'gitterroststufe'  # Standard-Tiefen nur als Gegencheck
            else:
                d['art'] = 'gitterrost'
        # v68: Rest-Logik laeuft jetzt im ALLE-TEILE-Nachlauf (das continue-Gate dieses
        #   Loops liess nur Rost/Stufen durch - nichtstahl/Zerlegung/Laschen waren tot)
        if d['art'] != alt_art:
            umsortiert += 1
    # ★ Stufenbreite = ZUKAUFMASS: Rostbreite + 2x 3-mm-Lasche - fuer JEDE Stufe genau einmal,
    #   unabhaengig davon, auf welchem Weg sie als Stufe erkannt wurde (1154 -> 1160)
    for d in teile.values():
        if isinstance(d, dict) and d.get('art') == 'gitterroststufe' and d.get('masse') and not d.get('_plus6'):
            d['_plus6'] = 1
            d['masse'][0] = round(d['masse'][0]) + 6
    if umsortiert:
        print('* Rost/Stufe nach Rostklasse umsortiert: %d Teile' % umsortiert)
    for d in teile.values():
        if isinstance(d, dict): d.pop('zentrum', None)
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
    # ★ Startzustand aus dem Plugin-Dialog (leer = Platzhalter bleibt = Standard)
    html = html.replace('__DEF_STAHL__', KONFIG.get('stahl') or '__DEF_STAHL__')
    html = html.replace('__DEF_GEL__', KONFIG.get('gelaender') or '__DEF_GEL__')
    html = html.replace('__GITTER_TEX__', KONFIG.get('gitter') or '__GITTER_TEX__')
    _dk = ';'.join('%s=%s' % (k, KONFIG[k]) for k in ('bws', 'min', 't1', 't2', 't3', 't4') if KONFIG.get(k))
    html = html.replace('__DEKOR__', _dk or '__DEKOR__')
    html = html.replace('__EXPIRY__', expiry)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    open(args.output, 'w', encoding='utf-8').write(html)
    print('OK: ' + args.output + ' (%d KB)' % (os.path.getsize(args.output) // 1024))
    try:
        with open(os.path.join(os.path.dirname(args.output), 'bericht.txt'), 'w', encoding='utf-8') as bf:
            bf.write('konverter=v68\nknick=breitenregel-26-8\nflaechen_gesamt=%d\nflaechen_leer=%d\nflaechen_unplanar_1mm=%d\ndoppelflaechen=%d\nteile_dicht=%d\nkoplanar_flaechen=%d\ndeckel_verworfen=%d\nlochdeckel=%d\n'
                     % (_FLSTAT['gesamt'], _FLSTAT['leer'], _FLSTAT['unplanar'], _FLSTAT.get('doppel', 0), _FLSTAT.get('dicht', 0), _FLSTAT.get('koplanar', 0), _FLSTAT.get('deckel', 0), _FLSTAT.get('lochdeckel', 0)))
            bf.write('gew_profil_stahl=%.2f\ngew_profil_gelaender=%.2f\ngew_blech_stahl=%.2f\ngew_blech_gelaender=%.2f\ngew_nichtstahl_ausgeschlossen=%.2f\n'
                     % (_FLSTAT.get('gw_prof', 0.0), _FLSTAT.get('gw_prof_gel', 0.0), _FLSTAT.get('gw_blech', 0.0), _FLSTAT.get('gw_blech_gel', 0.0), _FLSTAT.get('gw_nichtstahl', 0.0)))
        print('* Flaechen-Bericht: gesamt=%d leer=%d unplanar=%d (bericht.txt)'
              % (_FLSTAT['gesamt'], _FLSTAT['leer'], _FLSTAT['unplanar']))
    except Exception:
        pass

if __name__ == '__main__':
    main()
