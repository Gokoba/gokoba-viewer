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
# ★ v110 SCHALTER: die Entflechtung verschiebt echte Bauteile, um deckungsgleiche Flaechen
#   zu trennen. Sie ist AUS, weil der Versatz an Profilstoessen und am Bestand sichtbar war.
#   Auf True setzen holt die gedeckelte v109-Fassung zurueck (ein Durchlauf, 0,4 mm je Bauteil).
GK_ENTFLECHTEN = False
import numpy as np
import ifcopenshell, ifcopenshell.geom
import ifcopenshell.util.element as ue
import trimesh

# ★ Farbtabelle: AS-Layername → RGB. AS schreibt Umlaute als '?' in die IFC,
#   deshalb stehen die Namen hier genauso. Bei Bedarf anpassen/ergaenzen.
import time as _t74
_T0 = _t74.time()
KONFIG = {}
ACHSEN_ROH = []  # v70: Raster-Achsen aus direkt.json

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
    try:  # v90: Loecher, die AUSSERHALB der Aussenkontur liegen, verwerfen -
        #   Pauls Tuer-Modell: F1 der Daemmwand trug einen fremd zugeordneten
        #   Innenring komplett neben der eigenen Flaeche -> earcut-Muell.
        if loecher:
            _amn = np.asarray(aussen, dtype=float).min(axis=0) - 0.010
            _amx = np.asarray(aussen, dtype=float).max(axis=0) + 0.010
            _lok = []
            for _r in loecher:
                _b = np.asarray(_r, dtype=float)
                if len(_b) >= 3 and np.all(_b.min(axis=0) >= _amn) and np.all(_b.max(axis=0) <= _amx):
                    _lok.append(_r)
                else:
                    _FLSTAT['loch_aussen'] = _FLSTAT.get('loch_aussen', 0) + 1
            loecher = _lok
    except Exception:
        pass
    try:  # v89: Loch-Ausschnitte 2% zum Zentrum schrumpfen - schliesst die Phasen-Zwickel
        #   zwischen Ausschnittring (Stegflaeche) und Lochwand: die zwei getrennt erzeugten
        #   16-Ecke stehen verdreht, durch die Spalte war die Wand ringsum sichtbar
        #   (Pauls voller Kranz); 2% von d20 = 0,2 mm je Seite, optisch unsichtbar.
        if loecher:
            _l89 = []
            for _r in loecher:
                _a89 = np.asarray(_r, dtype=float)
                if len(_a89) >= 3:
                    _z89 = _a89.mean(axis=0)
                    _a89 = _z89 + (_a89 - _z89) * 0.98
                    _l89.append(_a89.tolist())
                else:
                    _l89.append(_r)
            loecher = _l89
    except Exception:
        pass
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

# ===================== v98 RUNDUNG =====================
# Advance Steel liefert Rundungen bereits TESSELLIERT und laesst sich nicht feiner
# stellen (FACETRES gemessen wirkungslos): ein Rohr ROR 30 kommt mit 16 Mantel-
# streifen, der Umriss ist ein 16-Eck und wird nah an der Kamera sichtbar kantig.
# Die Normalen sind in Ordnung - es ist echte Geometrie. Hier wird der AEUSSERE
# Mantelzylinder erkannt und auf mindestens 48 Segmente verfeinert; die neuen
# Punkte liegen EXAKT auf dem echten Kreis, es wird also nichts erfunden.
# Erkannt wird nur ueber die Mantelflaechen (Normale senkrecht zur Achse), damit
# Gehrungen und Deckel den Kreis nicht verfaelschen. Vielecke mit weniger als 12
# Ecken (Sechskantkoepfe, Achteckplatten) und bereits feine Rundungen ab 24
# Segmenten (Schrauben) bleiben unangetastet.
# SELBSTTEST: nach der Verfeinerung wird die Zahl der Randkanten und die
# Huellbox verglichen - wird irgendetwas schlechter, kommen die Originalflaechen
# zurueck. Am Echtpaket verwirft der Test 17 von 92 Teilen; die uebrigen 75
# werden nachweislich sauber feiner.

import collections

def _rd_hull(P):
    """Konvexe Huelle 2D (Andrew monotone chain)."""
    P=sorted(set(map(tuple,np.round(P,3))))
    if len(P)<3: return np.array(P,dtype=float)
    def halb(pts):
        h=[]
        for p in pts:
            while len(h)>=2 and (h[-1][0]-h[-2][0])*(p[1]-h[-2][1])-(h[-1][1]-h[-2][1])*(p[0]-h[-2][0])<=0: h.pop()
            h.append(p)
        return h
    return np.array(halb(P)[:-1]+halb(P[::-1])[:-1],dtype=float)

def _rd_zylinder(faces, tol=0.05):
    """Erkennt den AEUSSEREN Mantelzylinder eines Bauteils.
       Ausgewertet werden NUR Flaechen, deren Normale senkrecht zur Achse steht -
       das sind die Mantelstreifen. Schraege Schnittflaechen (Gehrung) und Deckel
       fallen dadurch von selbst heraus, und ihre Punkte verfaelschen den Kreis nicht."""
    pts=[]; kanten=[]
    for a,lo in faces:
        r=np.asarray(a,dtype=float)
        if len(r)<3: continue
        pts.append(r)
        for i in range(len(r)): kanten.append((r[i], r[(i+1)%len(r)]))
    if not pts: return None
    stimmen=collections.defaultdict(float); sammel=collections.defaultdict(list)
    for p,q in kanten:
        d=q-p; L=float(np.linalg.norm(d))
        if L<1e-6: continue
        d=d/L
        if d[0]<0 or (abs(d[0])<1e-9 and (d[1]<0 or (abs(d[1])<1e-9 and d[2]<0))): d=-d
        sch=tuple(np.round(d,3))
        stimmen[sch]+=L; sammel[sch].append(d*L)
    if not stimmen: return None
    # Die gerundete Richtung dient NUR dem Abstimmen. Als Achse wird der exakte
    #   laengengewichtete Mittelwert genommen - bei einem meterlangen Rohr wandert
    #   der projizierte Umriss sonst um bis zu 1 mm und der Kreis passt nicht mehr.
    sch=max(stimmen.items(),key=lambda kv:kv[1])[0]
    a=np.sum(np.array(sammel[sch]),axis=0); a=a/np.linalg.norm(a)

    h=np.array([1.,0,0]) if abs(a[0])<0.9 else np.array([0,1.,0])
    e0=np.cross(a,h); e0/=np.linalg.norm(e0); e1=np.cross(a,e0)
    # nur Mantelstreifen: Normale senkrecht zur Achse
    M=[]
    for p in pts:
        n=np.zeros(3)
        for i in range(len(p)): n+=np.cross(p[i],p[(i+1)%len(p)])
        ln=np.linalg.norm(n)
        if ln<1e-9: continue
        if abs(float((n/ln)@a))<0.08: M.append(p)
    if len(M)<12: return None
    P=np.vstack(M)
    Q=np.unique(np.round(np.column_stack([P@e0,P@e1]),2),axis=0)
    if len(Q)<12: return None
    # Aussenschale herausloesen: Rohre haben Mantel AUSSEN und INNEN
    c0=Q.mean(axis=0); rr=np.linalg.norm(Q-c0,axis=1)
    aus=Q[rr>rr.max()-max(0.4,0.06*rr.max())]
    if len(aus)<12: return None
    x,y=aus[:,0],aus[:,1]
    A=np.column_stack([x,y,np.ones(len(aus))]); b=x*x+y*y
    try: sol,*_=np.linalg.lstsq(A,b,rcond=None)
    except Exception: return None
    c=np.array([sol[0]/2.0,sol[1]/2.0]); r=float(np.sqrt(max(sol[2]+c@c,0.0)))
    if r<3.0: return None
    _rest=np.abs(np.linalg.norm(aus-c,axis=1)-r).max()
    if _rest>max(0.25,0.02*r): return None
    w=np.sort(np.unique(np.round(np.degrees(np.arctan2(aus[:,1]-c[1],aus[:,0]-c[0]))%360,1)))
    # dicht beieinanderliegende Winkel sind Rundungsdubletten, zusammenfassen
    ww=[w[0]]
    for x2 in w[1:]:
        if x2-ww[-1]>1.0: ww.append(x2)
    w=np.array(ww)
    N=len(w)
    if N<12 or N>=24: return None      # <12 = echtes Vieleck, >=24 = schon fein genug
    sch=np.diff(np.concatenate([w,[w[0]+360]]))
    # v102 GEMESSEN an E199 (Pauls kantiger Handlauf, 3713 mm langes RR o30x2): eine LUECKE in
    #   der Winkelverteilung heisst nicht, dass es kein Kreis ist - sie heisst nur, dass ein paar
    #   Punkte nicht ins Aussenband gefallen sind. Passt der Kreis sehr genau (E199: 0,046 mm
    #   Restfehler bei erlaubten 0,30), wird die Luecke deshalb toleriert. Am Echtpaket: 92 -> 94
    #   verfeinerte Rundteile, unveraendert 2 Ablehnungen, KEIN Verlust.
    if sch.max()>(4.0 if _rest<=0.10 else 1.8)*np.median(sch): return None
    return {'a':a,'e0':e0,'e1':e1,'c':c,'r':r,'N':N,'schritt':float(np.median(sch))}

def _rd_verfeinern(faces, zyl, ziel=48, kappe=4, tol=0.15):
    """Jede Sehnenkante auf dem Zylinder wird unterteilt, neue Punkte AUF dem echten Kreis."""
    a,e0,e1,c,r,N=zyl['a'],zyl['e0'],zyl['e1'],zyl['c'],zyl['r'],zyl['N']
    K=int(min(kappe,max(1,int(np.ceil(ziel/N)))))
    if K<2: return faces,0,K
    def lage(p):
        u=float(p@e0)-c[0]; v=float(p@e1)-c[1]
        return np.hypot(u,v), np.degrees(np.arctan2(v,u))%360, float(p@a)
    neu=[]; n=0
    for aus,lo in faces:
        rr=[]
        for ring in [np.asarray(aus,dtype=float)]+[np.asarray(x,dtype=float) for x in lo]:
            out=[]
            m=len(ring)
            for i in range(m):
                p=ring[i]; q=ring[(i+1)%m]
                out.append(p)
                r1,w1,z1=lage(p); r2,w2,z2=lage(q)
                if abs(r1-r)>tol or abs(r2-r)>tol: continue
                dw=(w2-w1+540)%360-180
                if abs(abs(dw)-zyl['schritt'])>0.35*zyl['schritt']: continue
                for k in range(1,K):
                    _f=k/float(K)
                    wk=np.radians(w1+dw*_f)
                    # v99: die Laengslage wird MITGEFUEHRT statt festgehalten. Vorher wurden nur
                    #   waagerechte Sehnen unterteilt; bei Gehrungen und schraeg abgeschnittenen
                    #   Rohren blieb die Deckelkante ungeteilt, dadurch entstanden T-Stoesse und
                    #   der Selbsttest hat das ganze Teil verworfen.
                    out.append((c[0]+r*np.cos(wk))*e0+(c[1]+r*np.sin(wk))*e1+(z1+(z2-z1)*_f)*a)
                    n+=1
            rr.append(np.array(out,dtype=float))
        neu.append((rr[0],rr[1:]))
    return neu,n,K

def _rd_randkanten(faces):
    """Kanten, die nur einmal vorkommen. Steigt die Zahl durch die Verfeinerung,
       sind Loecher oder T-Stoesse entstanden - dann wird verworfen."""
    s=collections.Counter()
    for a,lo in faces:
        for r in [a]+list(lo):
            r=np.round(np.asarray(r,dtype=float),3); m=len(r)
            if m<3: continue
            for i in range(m): s[tuple(sorted((tuple(r[i]),tuple(r[(i+1)%m]))))]+=1
    return sum(1 for v in s.values() if v==1)

def _rd_runden(faces, ziel=48, kappe=4):
    """Sichere Aussenhuelle: erkennen, _rd_verfeinern, PRUEFEN. Bei jedem Zweifel
       kommen die Original-Flaechen zurueck."""
    zy=_rd_zylinder(faces)
    if not zy: return faces,None
    neu,n,K=_rd_verfeinern(faces,zy,ziel=ziel,kappe=kappe)
    if n==0 or K<2: return faces,None
    if _rd_randkanten(neu)>_rd_randkanten(faces): return faces,'_rd_randkanten'   # SELBSTTEST
    A=np.vstack([np.asarray(a,dtype=float) for a,_ in faces])
    B=np.vstack([np.asarray(a,dtype=float) for a,_ in neu])
    if float(np.max(np.abs(np.concatenate([B.max(0)-A.max(0),A.min(0)-B.min(0)]))))>0.6:
        return faces,'bbox'                                           # SELBSTTEST
    return neu,{'N':zy['N'],'K':K,'r':zy['r'],'neu':n}

# =================== ENDE v98 RUNDUNG ===================

# ===================== v113 VOLLKOERPER-WEG =====================
# Pauls Hinweis, der den Knoten geloest hat: ueber den STEP-Weg gibt es diese
# Ueberstaende NIE - dort kommen echte VOLUMENKOERPER an, hier eine Flaechenliste.
# Genau daran lag es: Pauls Fassadenwand E2773 kam mit 80 offenen Kanten und
# 163 m Lochumfang an, war also gar kein geschlossener Koerper. Durch die Loecher
# sieht man die dahinterliegenden Flaechen - das ist der "Flaechenueberstand".
# Alle bisherigen Regeln (Tuerdeckel, Vollflaechen-Duplikat, Schalen-Ergaenzung)
# haben an genau diesem Symptom herumgeflickt, statt einen Koerper zu bauen.
#
# DER WEG: Bestandsteile sind achsparallele Bloecke. Aus allen vorkommenden
# x-, y- und z-Werten wird ein Raster aufgespannt; fuer jede Zelle sagt die
# VERALLGEMEINERTE WINDUNGSZAHL, ob sie im Material liegt - die vertraegt
# loechrige Netze, im Gegensatz zur Strahlenparitaet. Aus den gefuellten Zellen
# wird die Aussenhaut gebaut: garantiert geschlossen, ohne eine einzige
# Innenflaeche. Schiedsrichter bleibt das echte AS-Volumen (1 %).
#
# GEMESSEN an Pauls Balkon: 15 von 20 Bestandsteilen getroffen, die meisten auf
# 0,00 %, E2773 auf 0,13 %; alle 15 mit NULL offenen Kanten. Die uebrigen 5 liegen
# schief im Raum und fallen unveraendert auf den bisherigen Weg zurueck.
def _windungszahl113(P, T):
    w = np.zeros(len(P))
    for a, b, c in T:
        A = a - P; B = b - P; C = c - P
        la = np.linalg.norm(A, axis=1); lb = np.linalg.norm(B, axis=1); lc = np.linalg.norm(C, axis=1)
        num = np.einsum('ij,ij->i', A, np.cross(B, C))
        den = (la * lb * lc + np.einsum('ij,ij->i', A, B) * lc
               + np.einsum('ij,ij->i', B, C) * la + np.einsum('ij,ij->i', C, A) * lb)
        w += 2 * np.arctan2(num, den)
    return w / (4 * np.pi)

def _vollkoerper113(dreiecke, asvol):
    """Baut aus der Flaechenliste einen geschlossenen Vollkoerper. None = nicht anwendbar."""
    if not dreiecke or not asvol or float(asvol) <= 0: return None
    T = np.vstack(dreiecke).astype(float)
    if len(T) < 4 or len(T) > 20000: return None
    nv = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
    ln = np.linalg.norm(nv, axis=1)
    gut = ln > 1e-9
    if not gut.any(): return None
    _flA = ln / 2.0
    _e = np.zeros_like(nv); _e[gut] = nv[gut] / ln[gut][:, None]
    _schief = gut & (np.abs(_e).max(axis=1) < 0.9999)
    # ★ Ein Raster kann nur achsparallele Koerper abbilden. Eine WINZIGE Schraege darf
    #   trotzdem dabei sein - Pauls Fassadenwand E2773 hat oben an der Attika drei geneigte
    #   Dreiecke mit 0,46 % der Gesamtflaeche, und daran darf der ganze Koerper nicht
    #   scheitern. Ueber 1 % wird abgebrochen, damit kein schiefes Bauteil zur Treppe wird
    #   (genau der Fehler, den Paul am Daemmungsteil E3265 gefunden hat). Das echte
    #   AS-Volumen prueft danach ohnehin gegen.
    if _flA[_schief].sum() > 0.01 * _flA[gut].sum(): return None
    V = T.reshape(-1, 3)
    achsen = [np.unique(np.round(V[:, i], 4)) for i in range(3)]
    n = [len(a) - 1 for a in achsen]
    if min(n) < 1: return None
    if n[0] * n[1] * n[2] > 200000: return None
    mit = [(a[:-1] + a[1:]) / 2.0 for a in achsen]
    G = np.stack(np.meshgrid(*mit, indexing='ij'), axis=-1).reshape(-1, 3)
    voll = (_windungszahl113(G, T) > 0.5).reshape(n)
    if not voll.any(): return None
    d = [np.diff(a) for a in achsen]
    vol = float((voll * (d[0][:, None, None] * d[1][None, :, None] * d[2][None, None, :])).sum())
    if abs(vol - float(asvol)) > 0.01 * float(asvol): return None   # Schiedsrichter: echtes AS-Volumen
    # Aussenhaut: jede Grenzflaeche zwischen gefuellter und leerer Zelle
    P = np.pad(voll, 1, constant_values=False)
    ker = P[1:-1, 1:-1, 1:-1]
    quads = []
    for achse in (0, 1, 2):
        for seite, versch in ((1, -1), (0, 1)):
            frei = ~np.roll(P, versch, axis=achse)[1:-1, 1:-1, 1:-1]
            for i, j, k in zip(*np.where(ker & frei)):
                gr = [[achsen[0][i], achsen[0][i + 1]], [achsen[1][j], achsen[1][j + 1]],
                      [achsen[2][k], achsen[2][k + 1]]]
                w = gr[achse][seite]
                a1, a2 = [q for q in (0, 1, 2) if q != achse]
                ecken = []
                for u, v in ((0, 0), (1, 0), (1, 1), (0, 1)):
                    p = [0.0, 0.0, 0.0]; p[achse] = w; p[a1] = gr[a1][u]; p[a2] = gr[a2][v]
                    ecken.append(p)
                if seite == 0: ecken = ecken[::-1]
                quads.append(ecken)
    if not quads: return None
    Q = np.array(quads, dtype=float)
    tri = np.empty((2 * len(Q), 3, 3), dtype=float)
    tri[0::2] = Q[:, [0, 1, 2]]
    tri[1::2] = Q[:, [0, 2, 3]]
    return tri
# =================== ENDE v113 VOLLKOERPER-WEG ===================

# ===================== v97 WURZEL-WEG =====================
# Das Plugin (ab v115/v118) schneidet jedes BESTANDSTEIL mit fuenf Ebenen und schreibt
# die exakten Schnittkanten des echten CAD-Volumenkoerpers ins Paket:
#     D <achse> <min> <max> <lage>      Dicke-Achse und Schnittlage
#     S x1 y1 z1 x2 y2 z2               eine Schnittkante
# Daraus wird der Koerper NEU GEBAUT statt aus der lueckenhaften Flaechenliste geraten.
# Gemessen an Pauls Paket vom 27.07.: 16 von 16 Bestandsteilen eindeutig rekonstruierbar,
# alle Volumen auf 0,00 % genau, Flaechen auf gleicher Tiefe von 99,2 % auf 0,0 %.
# Teile OHNE D-Zeilen (aller Stahlbau) laufen unveraendert den alten Weg - der Riegel
# ist die blosse Existenz der Schnittdaten, die das Plugin nur fuer Bestand schreibt.

_WZ_BASIS = {0: (1, 2), 1: (2, 0), 2: (0, 1)}   # rechtshaendig: e0 x e1 = +Achse

def _wz_ringe(kanten, ax):
    """Schnittkanten zu geschlossenen Ringen verketten (2D in der Schnittebene)."""
    import collections as _c
    e0, e1 = _WZ_BASIS[ax]
    seg = []
    for k in kanten:
        a = (round(k[e0], 2), round(k[e1], 2)); b = (round(k[3 + e0], 2), round(k[3 + e1], 2))
        if a != b: seg.append((a, b))
    nach = _c.defaultdict(list)
    for i, (a, b) in enumerate(seg):
        nach[a].append((b, i)); nach[b].append((a, i))
    frei = set(range(len(seg))); out = []
    while frei:
        i0 = min(frei); a, b = seg[i0]; frei.discard(i0); weg = [a, b]; akt = b
        while True:
            nx = None
            for (p, i) in nach[akt]:
                if i in frei: nx = (p, i); break
            if nx is None: break
            frei.discard(nx[1]); akt = nx[0]; weg.append(akt)
            if akt == weg[0]: break
        if len(weg) > 3 and weg[0] == weg[-1]:
            out.append(np.array(weg[:-1], dtype=float))
    return out

def _wz_flaeche(p):
    x, y = p[:, 0], p[:, 1]
    return float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)) / 2.0

def _wz_drin(pt, poly):
    x, y = pt; n = len(poly); c = False
    for i in range(n):
        x1, y1 = poly[i]; x2, y2 = poly[(i + 1) % n]
        if ((y1 > y) != (y2 > y)) and (x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-30) + x1): c = not c
    return c

def _wz_auf_kante(p, poly, tol=0.5):
    for i in range(len(poly)):
        a = poly[i]; b = poly[(i + 1) % len(poly)]; ab = b - a; l2 = float(ab @ ab)
        if l2 < 1e-12: continue
        s = max(0.0, min(1.0, float((p - a) @ ab) / l2))
        if float(np.linalg.norm(p - (a + s * ab))) <= tol: return True
    return False

def _wz_ring_drin(klein, gross, tol=0.5):
    """randtolerant - die Ecken duerfen exakt AUF der Kante des groesseren Rings liegen"""
    innen = 0
    for p in klein:
        if _wz_drin(p, gross): innen += 1
        elif not _wz_auf_kante(p, gross, tol): return False
    return innen > 0

def _wz_gruppiere(rs):
    """Ringe zu (Aussenring CCW, [Loecher CW]) ueber die Verschachtelungstiefe."""
    n = len(rs); tiefe = [0] * n; fl = [abs(_wz_flaeche(r)) for r in rs]
    for i in range(n):
        for j in range(n):
            if i != j and fl[j] > fl[i] and _wz_drin(rs[i][0], rs[j]): tiefe[i] += 1
    ccw = lambda r: r if _wz_flaeche(r) > 0 else r[::-1]
    cw = lambda r: r if _wz_flaeche(r) < 0 else r[::-1]
    grp = []
    for i in range(n):
        if tiefe[i] % 2: continue
        loch = [cw(rs[j]) for j in range(n) if tiefe[j] == tiefe[i] + 1 and _wz_drin(rs[j][0], rs[i])]
        grp.append((ccw(rs[i]), loch))
    return grp

def _wz_auf3d(p2, ax, u):
    e0, e1 = _WZ_BASIS[ax]
    p = np.zeros((len(p2), 3)); p[:, e0] = p2[:, 0]; p[:, e1] = p2[:, 1]; p[:, ax] = u
    return p

def _wz_koerper(abschnitte, ax):
    """abschnitte: [(u_min, u_max, ringe2d)] -> Dreiecke (M,3,3) in mm, Normalen nach aussen.
       An der Naht zweier Abschnitte wird der eingeschlossene Bereich aus dem unteren
       Deckel ausgestanzt und der obere weggelassen - sonst liegen dort zwei Flaechen
       exakt aufeinander, und genau das ist die Flimmer-Ursache."""
    grp = [_wz_gruppiere(rs) for (_a, _b, rs) in abschnitte]
    n = len(abschnitte)
    zus_o = [[[] for _ in g] for g in grp]; zus_u = [[[] for _ in g] for g in grp]
    weg_u = [[False] * len(g) for g in grp]; weg_o = [[False] * len(g) for g in grp]
    # v104: Inseln an der Naht. Hat der anschliessende Abschnitt LOECHER (etwa Ankertaschen im
    #   Fundament), muss unter jedem dieser Loecher ein Deckel stehen - dort liegt ja Material.
    #   Bisher habe ich solche Naehte einfach uebersprungen, dann blieben BEIDE Deckel stehen und
    #   man sah sie als Kante mitten im Block. Genau das hat Paul am Treppenturm-Fundament gesehen.
    ins_o = [[[] for _ in g] for g in grp]; ins_u = [[[] for _ in g] for g in grp]
    for i in range(n - 1):
        if abs(abschnitte[i][1] - abschnitte[i + 1][0]) > 1e-6: continue
        for a, (au_a, lo_a) in enumerate(grp[i]):
            for b, (au_b, lo_b) in enumerate(grp[i + 1]):
                if _wz_ring_drin(au_b, au_a):
                    zus_o[i][a].append(au_b)
                    for _h104 in lo_b: ins_o[i][a].append(_h104)   # v104: Deckel unter der Tasche
                    weg_u[i + 1][b] = True
                elif _wz_ring_drin(au_a, au_b):
                    zus_u[i + 1][b].append(au_a)
                    for _h104 in lo_a: ins_u[i + 1][b].append(_h104)
                    weg_o[i][a] = True
    cw = lambda r: r if _wz_flaeche(r) < 0 else r[::-1]
    tris = []
    for i, (ua, ub, rs) in enumerate(abschnitte):
        if ub - ua <= 1e-9: continue
        for gi, (aussen, loecher) in enumerate(grp[i]):
            for u, soll, fort, zus, insel in ((ua, -1.0, weg_u[i][gi], zus_u[i][gi], ins_u[i][gi]),
                                              (ub, +1.0, weg_o[i][gi], zus_o[i][gi], ins_o[i][gi])):
                if fort: continue
                lo = loecher + [cw(np.asarray(z)) for z in zus]
                t = _flaeche_zerlegen_kern(_wz_auf3d(aussen, ax, u), [_wz_auf3d(h, ax, u) for h in lo])
                if t is None or len(t) == 0: continue
                t = np.asarray(t, dtype=float)
                if np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])[:, ax].sum() * soll < 0:
                    t = t[:, ::-1, :]      # Deckel messbar ausrichten, nicht raten
                tris.append(t)
                for _i104 in insel:        # v104: unter jeder Tasche ein eigener Deckel
                    _ti = _flaeche_zerlegen_kern(_wz_auf3d(np.asarray(_i104), ax, u), [])
                    if _ti is None or len(_ti) == 0: continue
                    _ti = np.asarray(_ti, dtype=float)
                    if np.cross(_ti[:, 1] - _ti[:, 0], _ti[:, 2] - _ti[:, 0])[:, ax].sum() * soll < 0:
                        _ti = _ti[:, ::-1, :]
                    tris.append(_ti)
            for ring in [aussen] + loecher:   # aussen CCW, Loecher CW -> Normalen nach aussen
                A = _wz_auf3d(ring, ax, ua); B = _wz_auf3d(ring, ax, ub); m = len(ring)
                for j in range(m):
                    k = (j + 1) % m
                    tris.append(np.array([[A[j], A[k], B[k]], [A[j], B[k], B[j]]]))
    return np.vstack(tris) if tris else None

def _wz_bauen(schnitte, fl_ringe, vol_as=None):
    """schnitte: [{'ax','min','max','lage','S'}] + die Rohkonturen des Teils.
       Rueckgabe: Dreiecke (M,3,3) in mm oder None (dann bleibt der alte Weg).
       ZWEI SELBSTPRUEFUNGEN, beide an Pauls Paket vom 27.07. gemessen (je 0 Verstoesse):
       A) Der Koerper muss laengs der Schnittachse ein Prisma sein - jede Flaeche also
          parallel ODER senkrecht zur Achse. Eine schraege Flaeche hiesse: die Kontur
          aendert sich innerhalb eines Abschnitts, dann darf nicht extrudiert werden.
       B) Jedes Intervall zwischen zwei Trennebenen muss von einer Schnittlage abgetastet
          sein. Sonst waere unbekannt, was darin steckt, und der Koerper verlore ein Stueck.
       Schlaegt eine der beiden an, liefert die Funktion None und der alte Facettenweg
       uebernimmt - lieber der bekannte Stand als ein still falscher Koerper."""
    if not schnitte: return None
    ax = int(schnitte[0]['ax'])
    if any(int(s['ax']) != ax for s in schnitte): return None
    # v102: Ist das echte Volumen aus Advance Steel bekannt, entscheidet AM ENDE der Vergleich
    #   mit ihm - und nicht mehr meine Heuristik. Gemessen an Pauls Balkonturm: die Wand E3254
    #   mit den fuenf Tueroeffnungen wurde von der Prismen-Pruefung VERWORFEN, obwohl die
    #   Rekonstruktion das AS-Volumen auf 0,02 % trifft. Genau diese Wand fiel dadurch auf den
    #   alten Facettenweg zurueck - das waren die ueberstehenden Flaechen an den Tueren.
    _mitVol = (vol_as is not None and vol_as > 0.0)
    # ★ v103 SELBSTPRUEFUNG A, NEU GEFASST UND IMMER AKTIV.
    #   In v102 hatte ich sie abgeschaltet, sobald das Volumen bekannt war - ein Fehler:
    #   ein SCHIEF im Raum liegendes Bauteil laesst sich nicht durch Aufeinanderstapeln von
    #   Quadern nachbauen, auch wenn das Volumen zufaellig stimmt. Genau das ist Paul beim
    #   Daemmungs-Bauteil E3265 passiert.
    #   Statt "eine einzige schraege Flaeche verbietet alles" zaehlt jetzt der FLAECHENANTEIL.
    #   Gemessen an Pauls Balkonturm trennt das sauber:
    #     E3265 schief 97,0 %   E3264 Sturz 85,7 %   -> ablehnen
    #     E3254 Tuerwand 0,4 %  alle uebrigen 0,0 %  -> bauen
    _fges = 0.0; _fschr = 0.0
    for _a97, _l97 in fl_ringe:
        _p97 = np.asarray(_a97, dtype=float)
        if len(_p97) < 3: continue
        _n97 = np.zeros(3)
        for _i97 in range(len(_p97)):
            _n97 += np.cross(_p97[_i97], _p97[(_i97 + 1) % len(_p97)])
        _ln97 = float(np.linalg.norm(_n97))
        if _ln97 < 1e-9: continue
        _fges += _ln97 * 0.5
        _c97 = abs(float(_n97[ax]) / _ln97)
        # v106 GEMESSEN an Pauls Treppenturm-Fundament E14391: dessen Oberseite ist um 1,18 Grad
        #   geneigt. Die alte Schwelle 0,98 liess alles bis rund 11 Grad als "waagerecht" durch -
        #   eine leicht geneigte Flaeche wurde dadurch in STUFEN zerlegt, statt das Teil abzulehnen.
        #   Genau das waren die Kanten im Fundament, die es in Wirklichkeit nicht gibt.
        #   0,9999 entspricht 0,8 Grad; darunter ist eine Flaeche wirklich waagerecht.
        #   Gegengerechnet: Treppenturm 26 -> 25 gebaute Teile (genau das geneigte Fundament
        #   faellt heraus), Treppe Ost unveraendert 16.
        if 0.02 < _c97 < 0.9999: _fschr += _ln97 * 0.5
    if _fges <= 0.0: return None
    if _fschr > 0.05 * _fges:
        _FLSTAT['wz_schraeg'] = _FLSTAT.get('wz_schraeg', 0) + 1
        return None
    for _a97, _l97 in ():                             # (alte punktweise Pruefung entfaellt)
        _p97 = np.asarray(_a97, dtype=float)
        if len(_p97) < 3: continue
        _n97 = np.zeros(3)
        for _i97 in range(len(_p97)):
            _n97 += np.cross(_p97[_i97], _p97[(_i97 + 1) % len(_p97)])
        _ln97 = float(np.linalg.norm(_n97))
        if _ln97 < 1e-9: continue
        _c97 = abs(float(_n97[ax]) / _ln97)
        if 0.001 < _c97 < 0.999: return None          # schraege Flaeche -> kein Prisma
    pkt = [np.asarray(a, dtype=float) for a, _l in fl_ringe if len(a) >= 3]
    pkt += [np.asarray(l, dtype=float) for _a, ls in fl_ringe for l in ls if len(l) >= 3]
    if not pkt: return None
    ebenen = np.unique(np.round(np.vstack(pkt)[:, ax], 1))
    if len(ebenen) < 2: return None
    # Soll-Raster der Schnittlagen. MUSS zu _lagen115 im Plugin passen; stimmt es nicht
    #   mit den tatsaechlich gelieferten Lagen ueberein, ist das Raster veraltet - dann
    #   wird Pruefung B stillgelegt statt fehlzuschlagen.
    _mn97 = float(schnitte[0]['min']); _mx97 = float(schnitte[0]['max'])
    _soll97 = [_mn97 + (_mx97 - _mn97) * f for f in (0.10, 0.30, 0.50, 0.70, 0.90)]
    _raster97 = all(any(abs(float(s['lage']) - r) < 1.0 for r in _soll97) for s in schnitte)
    # v101: Ab Plugin v123 schreibt der Exporter fuer JEDE versuchte Schnittlage eine D-Zeile -
    #   auch dann, wenn dort kein Material lag (dann ohne S-Zeilen). Damit ist unterscheidbar,
    #   ob ein Abschnitt LEER war oder gar nicht abgetastet wurde. Aeltere Pakete kennen das
    #   nicht; fuer die gilt weiter das feste Raster 10/30/50/70/90 Prozent.
    absch = []
    for a, b in zip(ebenen[:-1], ebenen[1:]):
        rs = None; getastet = False
        for s in schnitte:                       # STRENG innen - Lagen auf der Trennebene sind mehrdeutig
            if a + 0.5 < float(s['lage']) < b - 0.5:
                getastet = True
                rs = _wz_ringe(s['S'], ax) if s.get('S') else None
                break
        if rs:
            absch.append((float(a), float(b), rs))
        elif getastet:
            continue                             # abgetastet und leer -> korrekt uebersprungen
        elif _raster97 and any(a + 0.5 < r < b - 0.5 for r in _soll97):
            continue                             # altes Paket: Soll-Lage lag darin, also leer
        elif _mitVol:
            continue                             # v102: das Volumen entscheidet, nicht die Heuristik
        else:
            return None                          # SELBSTPRUEFUNG B: Abschnitt nie abgetastet
    if not absch: return None
    if len(absch) > 60: return None      # v103: Ausreisser abfangen, statt den Bau zu ersticken
    T = _wz_koerper(absch, ax)
    if T is None or len(T) < 4: return None
    if len(T) > 60000: return None       # v103: kein Teil braucht so viele Dreiecke
    vol = float(np.sum(np.einsum('ij,ij->i', T[:, 0], np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])))) / 6.0
    if vol <= 0.0: return None                   # Selbstkontrolle: ein Koerper hat positives Volumen
    if _mitVol:
        # 1 % Toleranz - das deckt Rundung und Vermaschung ab, faengt aber jede echte Entgleisung.
        if abs(vol - vol_as) > 0.01 * vol_as:
            _FLSTAT['wz_volumen'] = _FLSTAT.get('wz_volumen', 0) + 1
            return None
    return T
# =================== ENDE v97 WURZEL-WEG ===================


def _entflechten107(szene, spalt=0.4):
    """v107 ENTFLECHTUNG - die eigentliche Loesung des Flackerns.

    Gemessen an Pauls Treppe Ost: 314 Bauteilpaare haben EXAKT deckungsgleiche, sich
    ueberlappende Flaechen - 288 davon Sonderteil gegen Sonderteil und 26 Gelaender gegen
    Gelaender, also INNERHALB derselben Klasse. Eine Klassen-Rangfolge im Viewer kann solche
    Paare grundsaetzlich nicht trennen, beide bekommen denselben Wert. Genau deshalb hat keine
    der Rangfolgen geholfen, die ich gebaut habe.

    Zwei Flaechen auf exakt derselben Tiefe streiten sich beim Drehen immer. Die einzige
    verlaessliche Loesung ist, dass sie nicht mehr exakt gleich tief liegen. Dieser Durchlauf
    zieht deshalb beim KLEINEREN Teil die betroffene Flaeche um 0,4 mm nach innen. Das Bauteil
    bleibt geschlossen, wird an dieser einen Flaeche 0,4 mm kuerzer und ist damit dauerhaft
    eindeutig. 0,4 mm sind im Stahlbau nicht darstellbar und in keiner Ansicht sichtbar.

    Speicherschonend: je Ebene und Bauteil werden NUR die Grenzen gemerkt, nicht die Dreiecke.
    """
    import numpy as _np
    try:
        namen = list(szene.geometry.keys())
    except Exception:
        return 0
    if len(namen) < 2:
        return 0
    ebenen = {}
    box = {}
    for nm in namen:
        g = szene.geometry[nm]
        try:
            V = _np.asarray(g.vertices, dtype=float); T = _np.asarray(g.faces)
        except Exception:
            continue
        if len(T) == 0 or len(V) == 0: continue
        box[nm] = (V.min(axis=0), V.max(axis=0))
        P = V[T]
        nn = _np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0])
        ln = _np.linalg.norm(nn, axis=1)
        ok = ln > 1e-12
        if not ok.any(): continue
        nn = nn[ok] / ln[ok][:, None]
        Q = P[ok]
        fl = (nn[:, 0] < 0) | ((_np.abs(nn[:, 0]) < 1e-9) & ((nn[:, 1] < 0) |
             ((_np.abs(nn[:, 1]) < 1e-9) & (nn[:, 2] < 0))))
        nn[fl] *= -1
        dd = _np.einsum('ij,ij->i', nn, Q[:, 0])
        # v107b: Dreiecksnormalen sind gerechnet, nicht exakt - 3 Stellen reichen fuer
        #   'dieselbe Ebene', 4 Stellen waren zu streng und fanden fast nichts.
        schl = _np.column_stack([_np.round(nn, 3), _np.round(dd, 4)])
        lo = Q.min(axis=1); hi = Q.max(axis=1)
        for i in range(len(dd)):
            k = (schl[i, 0], schl[i, 1], schl[i, 2], schl[i, 3])
            e = ebenen.setdefault(k, {})
            v = e.get(nm)
            if v is None:
                e[nm] = [lo[i].copy(), hi[i].copy()]
            else:
                _np.minimum(v[0], lo[i], out=v[0]); _np.maximum(v[1], hi[i], out=v[1])
    getan = 0
    schon = set()   # ★ v109: jedes Bauteil darf hoechstens EINMAL verschoben werden
    for k, teile in ebenen.items():
        if len(teile) < 2: continue
        nvec = _np.array(k[:3], dtype=float); d0 = float(k[3])
        ks = sorted(teile)
        for i in range(len(ks)):
            for j in range(i + 1, len(ks)):
                a, b = ks[i], ks[j]
                A = teile[a]; B = teile[b]
                # v107c: NUR IN DER EBENE pruefen. Quer zur Ebene ist die Ausdehnung null,
                #   ein Test ueber alle drei Richtungen konnte deshalb nie zutreffen - genau
                #   deshalb fand der erste Anlauf so gut wie nichts.
                u = _np.minimum(A[1], B[1]) - _np.maximum(A[0], B[0])
                _in = _np.argsort(_np.abs(nvec))[:2]       # die beiden Richtungen IN der Ebene
                if not (u[_in[0]] > 0.0005 and u[_in[1]] > 0.0005):
                    continue
                va = float(_np.prod(box[a][1] - box[a][0] + 1e-9))
                vb = float(_np.prod(box[b][1] - box[b][0] + 1e-9))
                klein = a if va <= vb else b
                # ★ v109 DECKEL: ohne diese Sperre wurde dasselbe Bauteil in JEDER Ebene
                #   erneut verschoben, in der es einen Nachbarn hat, und das sechsmal
                #   hintereinander. Gemessen an Pauls Balkon: 433 von 1067 Bauteilen
                #   verschoben, groesster Weg 16,42 mm - genau die Spalte und Versatzmasse,
                #   die er an den Profilstoessen gesehen hat. Mit der Sperre bleibt es bei
                #   einmal 0,4 mm je Bauteil, und das ist in keiner Ansicht darstellbar.
                if klein in schon: continue
                # v107g GRUNDLEGEND ANDERS UND SICHER: nicht mehr die Flaeche verschieben,
                #   sondern das GANZE Bauteil. Das Verformen einzelner Flaechen hat Teile
                #   aufgerissen - Paul sah sie stellenweise durchsichtig. Ein starres Verschieben
                #   um 0,4 mm kann das nicht: die Form bleibt exakt erhalten, das Volumen auch,
                #   und die beiden Flaechen liegen trotzdem nicht mehr gleich tief.
                gk = szene.geometry[klein]
                gross = b if klein == a else a
                V = _np.asarray(gk.vertices, dtype=float)
                mk = (box[klein][0] + box[klein][1]) * 0.5
                mg = (box[gross][0] + box[gross][1]) * 0.5
                weg = 1.0 if float((mk - mg) @ nvec) >= 0 else -1.0   # vom Nachbarn WEG
                V += nvec * (weg * spalt * 0.001)
                box[klein] = (V.min(axis=0), V.max(axis=0))
                try:
                    gk.vertices = V
                except Exception:
                    continue
                getan += 1
                schon.add(klein)
    try:
        _FLSTAT['entflochten'] = getan
    except Exception:
        pass
    return getan


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
        if meta.get('achsen'):
            ACHSEN_ROH[:] = meta.get('achsen')  # v70
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
    kn = None; dreiecke = []; aussen = None; loecher = []; fl_ntris = []; fl_breitL = []; fl_hatLoch = []; fl_lochB = []; fl_outBB = []; fl_ringe = []

    def _fl_ab():
        nonlocal flLeer
        if aussen is not None:
            t3 = _flaeche_zerlegen(aussen, loecher)
            if t3 is not None:
                dreiecke.append(t3)
                fl_ntris.append(len(t3))
                fl_breitL.append(_FLSTAT.get('letzte_breite', 99.0) > 12.0)  # v56: >12mm = breite Ebene
                fl_hatLoch.append(bool(loecher))  # v58: Deckel-Erkennung
                _ao90 = np.asarray(aussen, dtype=float)
                fl_outBB.append((_ao90.min(axis=0), _ao90.max(axis=0)))  # v90: Tuer-Deckel-Paarung
                fl_ringe.append((list(aussen), [list(lo) for lo in (loecher or [])]))  # v96: Rohkonturen fuer die Schalen-Ergaenzung
                fl_lochB.append([(np.asarray(lo, dtype=float).min(axis=0), np.asarray(lo, dtype=float).max(axis=0)) for lo in (loecher or [])])  # v65: Lochring-Boxen
            else: flLeer += 1

    def _teil_ab():
        nonlocal n, fehler
        if kn is None: return
        try:
            if not dreiecke: return
            _istBest113 = any(_w in str((info.get(kn, {}) or {}).get('layer') or '').lower()
                              for _w in ('mmung', 'daemm', 'mauerwerk', 'beton', 'bestand', 'estrich'))
            # v97 WURZEL-WEG: Liegen fuer dieses Teil Schnittdaten aus dem CAD-Kern vor,
            #   wird der Koerper daraus NEU GEBAUT und ersetzt die Facetten vollstaendig.
            #   Damit entfaellt jedes Raten an der lueckenhaften Flaechenliste - und mit
            #   ihm die v96-Schalen-Ergaenzung, die nur ein Symptom geflickt hat.
            #   Nur EINE Flaeche mit Kennung 'breite Ebene' -> Kanten bleiben scharf,
            #   Waende flach (Schwelle 8 Grad statt 26).
            # ★ v113 VOLLKOERPER-WEG zuerst: liefert einen garantiert geschlossenen Koerper
            #   und macht damit jedes Flicken an der Flaechenliste ueberfluessig. Schlaegt er
            #   fehl (schief im Raum, Raster zu gross, Volumen passt nicht), bleibt alles
            #   genau wie bisher.
            _vk113 = None
            if _istBest113:
                try:
                    _vk113 = _vollkoerper113(dreiecke, (info.get(kn, {}) or {}).get('volumen'))
                except Exception:
                    _vk113 = None
            if _vk113 is not None:
                dreiecke[:] = [_vk113]
                fl_ntris[:] = [len(_vk113)]
                fl_breitL[:] = [True]
                fl_hatLoch[:] = [False]
                _bb113 = _vk113.reshape(-1, 3)
                fl_outBB[:] = [(_bb113.min(axis=0), _bb113.max(axis=0))]
                fl_lochB[:] = [[]]
                fl_ringe[:] = []          # schaltet Schalen-Ergaenzung und Duplikat-Regeln ab
                _FLSTAT['vollkoerper'] = _FLSTAT.get('vollkoerper', 0) + 1
            elif schnitte:
                _wzT = None
                try:
                    _wzT = _wz_bauen(schnitte, fl_ringe, (info.get(kn, {}) or {}).get('volumen'))
                except Exception:
                    _wzT = None
                if _wzT is not None:
                    dreiecke[:] = [_wzT]
                    fl_ntris[:] = [len(_wzT)]
                    fl_breitL[:] = [True]
                    fl_hatLoch[:] = [False]
                    _bb97 = _wzT.reshape(-1, 3)
                    fl_outBB[:] = [(_bb97.min(axis=0), _bb97.max(axis=0))]
                    fl_lochB[:] = [[]]
                    fl_ringe[:] = []          # schaltet die v96-Schalen-Ergaenzung ab
                    _FLSTAT['wurzelweg'] = _FLSTAT.get('wurzelweg', 0) + 1
                else:
                    _FLSTAT['wurzelweg_fehl'] = _FLSTAT.get('wurzelweg_fehl', 0) + 1
            # v98 RUNDUNG: Aussenzylinder feiner machen. Laeuft NACH dem Wurzel-Weg -
            #   Bestandsteile haben dort fl_ringe geleert und werden deshalb nicht angefasst.
            #   Die Zaehler in _FLSTAT werden um die Neu-Zerlegung herum gesichert, damit
            #   der Bericht weiter die echten Flaechenzahlen nennt.
            if fl_ringe and len(fl_ringe) >= 12:
                try:
                    _f98 = [(np.asarray(_a, dtype=float), [np.asarray(_h, dtype=float) for _h in _ls])
                            for _a, _ls in fl_ringe]
                    _nf98, _st98 = _rd_runden(_f98)
                except Exception:
                    _st98 = None
                if isinstance(_st98, dict):
                    _sich98 = {_k: _FLSTAT.get(_k, 0) for _k in
                               ('gesamt', 'leer', 'unplanar', 'loch_aussen')}
                    _dr98 = []; _nt98 = []; _fb98 = []; _fh98 = []; _fo98 = []; _fl98 = []; _fr98 = []
                    for _a98, _lo98 in _nf98:
                        _t98 = _flaeche_zerlegen(_a98.tolist(), [_x.tolist() for _x in _lo98])
                        if _t98 is None or len(_t98) == 0: continue
                        _dr98.append(_t98); _nt98.append(len(_t98))
                        _fb98.append(_FLSTAT.get('letzte_breite', 99.0) > 12.0)
                        _fh98.append(bool(_lo98))
                        _fo98.append((_a98.min(axis=0), _a98.max(axis=0)))
                        _fl98.append([(_x.min(axis=0), _x.max(axis=0)) for _x in _lo98])
                        _fr98.append((_a98.tolist(), [_x.tolist() for _x in _lo98]))
                    _FLSTAT.update(_sich98)
                    if len(_dr98) == len(fl_ringe):      # nur uebernehmen, wenn KEINE Flaeche verloren ging
                        dreiecke[:] = _dr98; fl_ntris[:] = _nt98; fl_breitL[:] = _fb98
                        fl_hatLoch[:] = _fh98; fl_outBB[:] = _fo98; fl_lochB[:] = _fl98
                        fl_ringe[:] = _fr98
                        _FLSTAT['rund'] = _FLSTAT.get('rund', 0) + 1
                    else:
                        _FLSTAT['rund_verworfen'] = _FLSTAT.get('rund_verworfen', 0) + 1
            try:  # v96: FEHLENDE AUSSENSCHALE ERGAENZEN.
                #   MESSUNG an Pauls Echtpaket: die Daemmwand E1065 (125 mm) liefert im Export
                #   NUR ihre Rueckseite (y=0, 36,6 m2 mit 8 Oeffnungen) - die AUSSENHAUT bei
                #   y=-125 fehlt vollstaendig. Der Betrachter sieht deshalb die Rueckseite, die
                #   exakt in derselben Ebene liegt wie die Mauerwerks-Vorderseite (y=0): zwei
                #   Flaechen auf identischer Tiefe = Flimmern und scheinbar 'erhoehte' Baender
                #   rings um Fenster und Tuer. Eine Wandschale hat zwingend ZWEI parallele
                #   Seiten; fehlt eine, wird sie aus der vorhandenen Kontur samt Oeffnungen
                #   exakt auf die Gegenebene kopiert - keine Schaetzung, dieselben Koordinaten.
                _ly96 = str((info.get(kn, {}) or {}).get('layer') or '').lower()
                if any(w in _ly96 for w in ('mmung', 'daemm', 'mauerwerk', 'beton', 'bestand', 'estrich')) and fl_ringe:
                    _eb96 = []
                    for _i96, (_a96, _l96) in enumerate(fl_ringe):
                        _p96 = np.asarray(_a96, dtype=float)
                        if len(_p96) < 3: continue
                        _g96 = _p96.max(axis=0) - _p96.min(axis=0)
                        _ax96 = int(np.argmin(_g96))
                        if _g96[_ax96] > 1.0: continue          # nicht eben
                        _fl96 = float(_g96[(_ax96 + 1) % 3] * _g96[(_ax96 + 2) % 3])
                        _eb96.append((_ax96, float(_p96[:, _ax96].mean()), _i96, _fl96))
                    for _ax96, _wert96, _i96, _flg96 in sorted(_eb96, key=lambda e: -e[3]):
                        if not fl_ringe[_i96][1]: continue      # Referenz muss die Oeffnungen tragen
                        _alle96 = np.vstack([np.asarray(_a, dtype=float) for _a, _ in fl_ringe if len(_a) >= 3])
                        _kand96 = [float(_k) for _k in np.unique(np.round(_alle96[:, _ax96], 1))
                                   if 5.0 < abs(float(_k) - _wert96) < 1000.0]
                        if not _kand96: break
                        _geg96 = min(_kand96, key=lambda k: abs(k - _wert96))
                        # ★ v111 WURZEL DES FEHLERS, an Pauls Balkon gemessen: die Gegenebene
                        #   wurde als der NAECHSTGELEGENE Koordinatenwert gewaehlt. Die Fassadenwand
                        #   E2773 hat aber vier Ebenen (2735,6 / 2906,8 / 3085,6 / 3435,6 mm) - die
                        #   echte Rueckseite liegt bei 3085,6, der naechste Wert war 2906,8. Damit
                        #   wurde eine 60 m2 grosse Schale 178,8 mm ZU WEIT VORN eingezogen, und ueberall
                        #   dort, wo die Wand dort duenner ist (Laibungen, Ruecksprunge), stand sie
                        #   ueber - genau die Flaechenueberstaende am Tuerdurchgang.
                        #   Und die alte Abbruchpruefung konnte das nicht fangen: sie sah NUR auf die
                        #   bereits falsch gewaehlte Ebene _geg96, nicht darauf, ob die Wand ueberhaupt
                        #   schon eine Rueckseite hat. Jetzt zuerst diese Frage - liegt auf IRGENDEINER
                        #   anderen Ebene derselben Achse eine parallele Flaeche von mindestens halber
                        #   Groesse, ist die Schale vollstaendig und es wird nichts ergaenzt.
                        if any(_a2 == _ax96 and abs(_w2 - _wert96) > 1.0 and _f2 > 0.5 * _flg96
                               for _a2, _w2, _j2, _f2 in _eb96): break   # Rueckseite ist da - nichts tun
                        if any(_a2 == _ax96 and abs(_w2 - _geg96) < 1.0 and _f2 > 0.5 * _flg96
                               for _a2, _w2, _j2, _f2 in _eb96): break   # Gegenseite ist da - nichts tun
                        _vs96 = _geg96 - _wert96
                        _au96 = np.asarray(fl_ringe[_i96][0], dtype=float).copy(); _au96[:, _ax96] += _vs96
                        _lo96 = []
                        for _r96 in fl_ringe[_i96][1]:
                            _rr96 = np.asarray(_r96, dtype=float).copy(); _rr96[:, _ax96] += _vs96
                            _lo96.append(_rr96.tolist())
                        _t96 = _flaeche_zerlegen(_au96[::-1].tolist(), _lo96)
                        if _t96 is not None and len(_t96):
                            dreiecke.append(_t96); fl_ntris.append(len(_t96))
                            fl_breitL.append(True); fl_hatLoch.append(True)
                            fl_outBB.append((_au96.min(axis=0), _au96.max(axis=0)))
                            fl_lochB.append([(np.asarray(_l).min(axis=0), np.asarray(_l).max(axis=0)) for _l in _lo96])
                            fl_ringe.append((_au96.tolist(), _lo96))
                            _FLSTAT['schale_ergaenzt'] = _FLSTAT.get('schale_ergaenzt', 0) + 1
                        break
            except Exception:
                pass
            t3 = np.vstack(dreiecke) * skal
            va = t3.reshape(-1, 3)
            fa = np.arange(len(va), dtype=np.int64).reshape(-1, 3)
            m = trimesh.Trimesh(vertices=va, faces=fa, process=False)
            m.merge_vertices()
            fl_id = np.repeat(np.arange(len(fl_ntris)), fl_ntris) if fl_ntris else None
            fl_breit = np.array(fl_breitL, dtype=bool) if fl_breitL else None
            try:  # v90: TUER-DECKEL: eine Flaeche, deren Aussenkontur deckungsgleich mit einem
                #   OEFFNUNGS-Loch einer ANDEREN Flaeche desselben Teils ist (parallel versetzte
                #   Wandseiten!), ist der Deckel, der die Oeffnung verschliesst -> verwerfen.
                _ly91 = str((info.get(kn, {}) or {}).get('layer') or '').lower()
                _istDaemm91 = any(w in _ly91 for w in ('mmung', 'daemm', 'mauerwerk', 'beton', 'bestand'))
                if _istDaemm91 and fl_id is not None and fl_outBB and any(fl_lochB):
                    _weg90 = np.zeros(len(fl_ntris), dtype=bool)
                    for _i90, (_omn, _omx) in enumerate(fl_outBB):
                        _tref = False
                        for _j90, _lbs in enumerate(fl_lochB):
                            if _j90 == _i90 or not _lbs: continue
                            for _lmn, _lmx in _lbs:
                                _lmn = np.asarray(_lmn, dtype=float); _lmx = np.asarray(_lmx, dtype=float)
                                _lgr = np.sort(_lmx - _lmn)[-2:]
                                if _lgr[0] < 400.0:
                                    continue  # v91: nur GROSSE Oeffnungen (Tuer/Fenster) - v90 fraß mit
                                              #   539 Treffern massenhaft Schraubenloch-Umgebung
                                _dz = np.abs(((_omn + _omx) - (_lmn + _lmx)) / 2.0)  # Zentren-Abstand, roh (mm)
                                _dg = np.abs((_omx - _omn) - (_lmx - _lmn))          # Groessen-Differenz (mm)
                                # deckungsgleich in den 2 grossen Achsen, ECHTER Versatz (>10mm) in der Dickenrichtung
                                _ok = (np.sort(_dz)[:2] < 25.0).all() and (np.sort(_dg)[:2] < 50.0).all() and (np.sort(_dz)[2] > 10.0)
                                if _ok:
                                    _tref = True; break
                            if _tref: break
                        if _tref:
                            _weg90[_i90] = True
                            _FLSTAT['tuerdeckel'] = _FLSTAT.get('tuerdeckel', 0) + 1
                    # v94: VOLLFLAECHEN-DUPLIKAT: der Export liefert Wandseiten MEHRFACH -
                    #   einmal MIT Oeffnungs-Loechern, zusaetzlich als ungelochte Vollflaechen
                    #   (Pauls Mauerwerkswand E4318: Fenster dadurch zu). Regel: ungelochte
                    #   Flaeche, koplanar mit einer GELOCHTEN Flaeche aehnlicher Groesse
                    #   desselben Teils -> Duplikat verwerfen, die gelochte gewinnt.
                    for _i94, (_omn, _omx) in enumerate(fl_outBB):
                        if _weg90[_i94] or (len(fl_hatLoch) > _i94 and fl_hatLoch[_i94]): continue
                        _og = _omx - _omn
                        _ax94 = int(np.argmin(_og))
                        if _og[_ax94] > 2.0: continue
                        for _j94, (_jmn, _jmx) in enumerate(fl_outBB):
                            if _j94 == _i94 or _weg90[_j94]: continue
                            if not (len(fl_hatLoch) > _j94 and fl_hatLoch[_j94]): continue
                            _jg = _jmx - _jmn
                            if int(np.argmin(_jg)) != _ax94 or _jg[_ax94] > 2.0: continue
                            if abs(float(_omn[_ax94] - _jmn[_ax94])) > 2.0: continue
                            _dz94 = np.abs((_omn + _omx) - (_jmn + _jmx)) / 2.0
                            _dg94 = np.abs(_og - _jg)
                            # ★ v112: die beiden GROSSEN Achsen werden jetzt namentlich geprueft.
                            #   Vorher stand hier np.sort(...)[:2] - die zwei KLEINSTEN Werte. Bei einer
                            #   ebenen Flaeche ist die Dickenachse aber immer 0, also blieb effektiv nur
                            #   EINE grosse Achse uebrig: Uebereinstimmung in einer Richtung genuegte, um
                            #   eine Flaeche als Duplikat wegzuwerfen. An Pauls Balkon hat das die
                            #   Wandflaeche 6693..7023 mm (6,17 m2) der Fassade E2773 verschluckt, weil sie
                            #   dieselbe Hoehe hat wie die grosse gelochte Flaeche daneben - obwohl sie
                            #   2425 mm daneben liegt und nur ein Zehntel so breit ist. Dort fehlte danach
                            #   die Aussenhaut, und man sah die dahinterliegenden Flaechen ueberstehen.
                            _and94 = [_q for _q in (0, 1, 2) if _q != _ax94]
                            if all(_dz94[_q] < 400.0 for _q in _and94) and all(_dg94[_q] < 800.0 for _q in _and94):
                                _weg90[_i94] = True
                                _FLSTAT['voll_duplikat'] = _FLSTAT.get('voll_duplikat', 0) + 1
                                break
                    # v92: DOPPEL-FACETTEN-Rasur (Pauls Boden 163% doppelt trianguliert = Flimmern):
                    #   koplanare kleinere Flaeche, die vollstaendig IN einer groesseren Flaeche
                    #   derselben Ebene desselben Daemm-Teils liegt, ist Doppel-Geometrie -> weg.
                    for _i92, (_omn, _omx) in enumerate(fl_outBB):
                        if _weg90[_i92]: continue
                        _og = _omx - _omn
                        _ax92 = int(np.argmin(_og))
                        if _og[_ax92] > 2.0: continue  # nur ebene (achsparallele) Flaechen
                        for _j92, (_jmn, _jmx) in enumerate(fl_outBB):
                            if _j92 == _i92 or _weg90[_j92]: continue
                            _jg = _jmx - _jmn
                            if int(np.argmin(_jg)) != _ax92 or _jg[_ax92] > 2.0: continue
                            if abs(float(_omn[_ax92] - _jmn[_ax92])) > 2.0: continue  # koplanar?
                            if not (np.all(_omn >= _jmn - 5.0) and np.all(_omx <= _jmx + 5.0)): continue
                            if float(np.prod(np.sort(_jg)[-2:])) <= 1.5 * float(np.prod(np.sort(_og)[-2:])): continue
                            _weg90[_i92] = True
                            _FLSTAT['doppel_facette'] = _FLSTAT.get('doppel_facette', 0) + 1
                            break
                    if _weg90.any():
                        _mk90 = ~_weg90[fl_id]
                        m.update_faces(_mk90)
                        fl_id = fl_id[_mk90]
            except Exception:
                pass
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
                                if len(_FLSTAT.setdefault('deckelkill_probe', [])) < 6:  # v73: Beweis (fehlende Flansche jagen)
                                    _p2 = pts2
                                    _FLSTAT['deckelkill_probe'].append('%.2fm2|%.0f,%.0f,%.0f' % (
                                        fl_area[fid], (_p2.min(axis=0)[0] + _p2.max(axis=0)[0]) * 500,
                                        (_p2.min(axis=0)[1] + _p2.max(axis=0)[1]) * 500,
                                        (_p2.min(axis=0)[2] + _p2.max(axis=0)[2]) * 500))
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
                            if float(fs[ax]) > 0.003:
                                continue  # v69: Deckel sind FLACH - dicke Kandidaten (Lochwand-Segmente,
                                          #   Stegdicke!) werden nie gekillt (Pauls voller-Ring-Befund)
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
                                if len(_FLSTAT.setdefault('lochdeckel_probe', [])) < 6:  # v69: Beweis im bericht
                                    _FLSTAT['lochdeckel_probe'].append('%.0f,%.0f,%.0f|%.1f,%.1f,%.1f' % (
                                        fc[0] * 1000, fc[1] * 1000, fc[2] * 1000, fs[0] * 1000, fs[1] * 1000, fs[2] * 1000))
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
    nT = nL = nH = 0; nD = nS = 0; flLeer = 0; probeZeilen = []
    schnitte = []
    with open(geo_pfad, encoding='utf-8', errors='replace') as fh:
        for zeile in fh:
            try:
                z = zeile.strip()
                if not z: continue
                if z[0] == 'T':
                    nT += 1
                    _fl_ab(); _teil_ab()
                    kn = z.split()[1] if len(z.split()) > 1 else None
                    dreiecke = []; aussen = None; loecher = []; fl_ntris = []; fl_breitL = []; fl_hatLoch = []; fl_lochB = []; fl_outBB = []; fl_ringe = []
                    schnitte = []   # v97 Wurzel-Weg: Schnittdaten dieses Teils
                elif z[0] == 'D':   # v97: "D <achse> <min> <max> <lage>"
                    nD += 1
                    _w97 = z.split()
                    if len(_w97) >= 5:
                        schnitte.append({'ax': int(_w97[1]), 'min': float(_w97[2]),
                                         'max': float(_w97[3]), 'lage': float(_w97[4]), 'S': []})
                elif z[0] == 'S':   # v97: "S x1 y1 z1 x2 y2 z2" - gehoert zur letzten D-Zeile
                    nS += 1
                    if schnitte:
                        _w97 = z.split()
                        if len(_w97) >= 7: schnitte[-1]['S'].append([float(x) for x in _w97[1:7]])
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
    print('* DIREKT-Diagnose: T=%d L=%d H=%d D=%d S=%d | Flaechen ohne Zerlegung: %d | unlesbare Zeilen: %d' % (nT, nL, nH, nD, nS, flLeer, kaputt))
    print('* VOLLKOERPER v113: %d Bestandsteile als geschlossener Koerper gebaut' % _FLSTAT.get('vollkoerper', 0))
    print('* WURZEL-WEG: %d Teile aus den CAD-Schnitten neu gebaut, %d mit Schnittdaten aber ohne Erfolg (alter Weg)'
          % (_FLSTAT.get('wurzelweg', 0), _FLSTAT.get('wurzelweg_fehl', 0)))
    # v107d: MEHRFACH laufen lassen. Wird eine Flaeche verschoben, kann sie auf ein drittes
    #   Bauteil treffen - ein einzelner Durchgang loest deshalb nur einen Teil. Nach jedem
    #   Durchgang wird gezaehlt; bleibt nichts mehr uebrig oder aendert sich nichts, ist Schluss.
    # ★ v108 WICKLUNG RICHTIGSTELLEN - die eigentliche Ursache des "Durchschauens".
    #   Gemessen an Pauls Treppe Ost: 155 von 818 Bauteilen haben eine nach INNEN gerichtete
    #   Flaechenwicklung, darunter U240-Wangen und FL-40x8-Untergurte. Die Grafikkarte schneidet
    #   rueckwaertige Flaechen weg - bei falscher Wicklung ist das ausgerechnet die AUSSENhaut.
    #   Man schaut dann in das Bauteil hinein und sieht, was dahinter liegt. Genau Pauls Befund:
    #   "die aeussere Wangenflaeche fehlt" und "die obere Flaeche vom Flachstahl fehlt".
    #   Pruefung ist das vorzeichenbehaftete Volumen: ist es negativ, zeigen die Flaechen nach
    #   innen und alle Dreiecke des Bauteils werden umgedreht. Ein geschlossener Koerper hat
    #   immer positives Volumen - der Test ist damit eindeutig und ohne Ermessensspielraum.
    _drehGes108 = 0
    try:
        for _nm108 in list(szene.geometry.keys()):
            _g108 = szene.geometry[_nm108]
            try:
                _T108 = np.asarray(_g108.triangles, dtype=float)
            except Exception:
                continue
            if len(_T108) < 4: continue
            # ★ v109 BEZUGSPUNKT: das vorzeichenbehaftete Volumen wird um die BAUTEILMITTE
            #   gerechnet, nicht mehr um den Weltursprung. Nur bei vollstaendig geschlossenen
            #   Koerpern ist der Wert lageunabhaengig; unsere Bauteile sind aber offene Schalen
            #   (Vertices gesplittet, einzelne Deckel fehlen). Dort haengt das Vorzeichen an der
            #   Modelllage - gemessen an Pauls Balkon hat der alte Test 264 KORREKT gewickelte
            #   Bauteile umgedreht (Traeger RR 30x20x2, Hutmuttern, Unterlegscheiben), waehrend
            #   der Test um die Bauteilmitte NULL Teile beanstandet. Gegenprobe an 50 kuenstlich
            #   umgedrehten Teilen: Bauteilmitte erkennt 50/50, Weltursprung nur 34/50.
            _m108 = (_T108.reshape(-1, 3).min(axis=0) + _T108.reshape(-1, 3).max(axis=0)) * 0.5
            _T108 = _T108 - _m108
            _v108 = float(np.sum(np.einsum('ij,ij->i', _T108[:, 0],
                          np.cross(_T108[:, 1] - _T108[:, 0], _T108[:, 2] - _T108[:, 0])))) / 6.0
            if _v108 < -1e-12:
                _f108 = np.asarray(_g108.faces).copy()
                _f108 = _f108[:, ::-1]
                _g108.faces = _f108
                _drehGes108 += 1
    except Exception:
        pass
    _FLSTAT['gedreht'] = _drehGes108
    print('* WICKLUNG v110: %d Bauteile nach innen gewickelt und umgedreht (Bezug: Bauteilmitte)' % _drehGes108)

    # ★ v110 ENTFLECHTUNG ABGESCHALTET. Auch der gedeckelte Rest war noch zu viel: gemessen
    #   an Pauls Balkon verschob sie 504 Bauteile, darunter 38 Traeger-/Gelaenderprofile
    #   (sichtbarer Restversatz an den Stoessen) und 18 der 20 BESTANDSTEILE - eine Betondecke
    #   0,4 mm vor ihrer Betonwand ist genau die "ueberstehende Flaeche", die Paul gemeldet hat.
    #   Ein Anzeigeproblem darf nicht durch das Verschieben echter Geometrie geloest werden;
    #   das Modell steht jetzt wieder bitgenau so da, wie Advance Steel es liefert.
    #   RUECKNAHME: GK_ENTFLECHTEN oben in dieser Datei auf True setzen (dann wieder ein
    #   Durchlauf mit Deckel je Bauteil, so wie in v109).
    # ★ v109: EIN Durchlauf. Die sechs Wiederholungen waren der zweite Teil des Schadens -
    #   ein verschobenes Bauteil erzeugte neue Ueberdeckungen und wurde in der naechsten Runde
    #   weitergeschoben; der Vorgang kam nie zur Ruhe (4113 "Trennungen" bei nur 504 wirklich
    #   vorhandenen Paaren). Ein Durchlauf mit Deckel trennt alle 504 Paare und bewegt nichts
    #   weiter als 0,4 mm.
    _entGes107 = _entflechten107(szene) if GK_ENTFLECHTEN else 0
    _FLSTAT['entflochten'] = _entGes107
    print('* ENTFLECHTUNG v110: %s (%d Bauteile verschoben) - Geometrie bleibt wie in Advance Steel'
          % ('AN' if GK_ENTFLECHTEN else 'AUS', _FLSTAT.get('entflochten', 0)))
    print('* WURZEL-WEG Formpruefung: %d Teile verworfen, weil sie schief im Raum liegen' % _FLSTAT.get('wz_schraeg', 0))
    print('* WURZEL-WEG Volumenpruefung: %d Teile verworfen, weil das Ergebnis nicht zum AS-Volumen passte' % _FLSTAT.get('wz_volumen', 0))
    print('* RUNDUNG: %d Bauteile feiner tesselliert, %d vom Selbsttest verworfen (Original behalten)'
          % (_FLSTAT.get('rund', 0), _FLSTAT.get('rund_verworfen', 0)))
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
        if meta.get('achsen'):
            ACHSEN_ROH[:] = meta.get('achsen')  # v70
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
        try:  # v78: BG-Namen NUR aus HAUPTTEILEN (eigene Pos == BG-Nr) - die Namensliste
            #   kreuzte Teil-Positionsnummern mit Baugruppen-Nummern (Pauls 'Riegel'-Beweis:
            #   Hauptteil der BG 2 traegt 'Stuetze', ein TEIL mit Pos 2 traegt 'Riegel')
            _bg78 = {}
            for _d in teile.values():
                if isinstance(_d, dict) and _d.get('bgnr') and _d.get('ref') and str(_d['bgnr']) == str(_d['ref']):
                    _a78 = (_d.get('attrs') or [None])[0]
                    if _a78:
                        _bg78[str(_d['bgnr'])] = str(_a78)
            if _bg78:
                namen_bg = _bg78
        except Exception:
            pass
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
        if d.get('art') in ('gitterrost', 'gitterroststufe'):
            d['gewicht'] = None  # v74: Roste/Stufen NIE mit Gewicht (Pauls Vorgabe - nirgendwo im Viewer)
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
    # v105: die Konverter-Version in den Viewer schreiben, damit sie ohne bericht.txt
    #   nachschlagbar ist (im Quelltext nach KONVERTER_V suchen).
    html = html.replace('__KONV__', 'V113')
    # ★ Startzustand aus dem Plugin-Dialog (leer = Platzhalter bleibt = Standard)
    import json as _j70
    html = html.replace("JSON.parse('__ACHSEN__')", _j70.dumps(ACHSEN_ROH) if ACHSEN_ROH else "null")  # v70: rohes Array-Literal
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
            bf.write('konverter=v113\nknick=breitenregel-26-8\nflaechen_gesamt=%d\nflaechen_leer=%d\nflaechen_unplanar_1mm=%d\ndoppelflaechen=%d\nteile_dicht=%d\nkoplanar_flaechen=%d\ndeckel_verworfen=%d\nlochdeckel=%d\n'
                     % (_FLSTAT['gesamt'], _FLSTAT['leer'], _FLSTAT['unplanar'], _FLSTAT.get('doppel', 0), _FLSTAT.get('dicht', 0), _FLSTAT.get('koplanar', 0), _FLSTAT.get('deckel', 0), _FLSTAT.get('lochdeckel', 0)))
            bf.write('gew_profil_stahl=%.2f\ngew_profil_gelaender=%.2f\ngew_blech_stahl=%.2f\ngew_blech_gelaender=%.2f\ngew_nichtstahl_ausgeschlossen=%.2f\n'
                     % (_FLSTAT.get('gw_prof', 0.0), _FLSTAT.get('gw_prof_gel', 0.0), _FLSTAT.get('gw_blech', 0.0), _FLSTAT.get('gw_blech_gel', 0.0), _FLSTAT.get('gw_nichtstahl', 0.0)))
            bf.write('lochdeckel_probe=%s\n' % ';'.join(_FLSTAT.get('lochdeckel_probe', [])))
            bf.write('loch_aussen=%d\ntuerdeckel=%d\ndoppel_facette=%d\nvoll_duplikat=%d\nschale_ergaenzt=%d\nwurzelweg=%d\nwurzelweg_fehl=%d\nrund=%d\nrund_verworfen=%d\n' % (_FLSTAT.get('loch_aussen', 0), _FLSTAT.get('tuerdeckel', 0), _FLSTAT.get('doppel_facette', 0), _FLSTAT.get('voll_duplikat', 0), _FLSTAT.get('schale_ergaenzt', 0), _FLSTAT.get('wurzelweg', 0), _FLSTAT.get('wurzelweg_fehl', 0), _FLSTAT.get('rund', 0), _FLSTAT.get('rund_verworfen', 0)))
            bf.write('deckelkill_probe=%s\n' % ';'.join(_FLSTAT.get('deckelkill_probe', [])))
            bf.write('dauer_konverter_s=%.1f\n' % (_t74.time() - _T0))  # v74: wo stecken die Minuten?
        print('* Flaechen-Bericht: gesamt=%d leer=%d unplanar=%d (bericht.txt)'
              % (_FLSTAT['gesamt'], _FLSTAT['leer'], _FLSTAT['unplanar']))
    except Exception:
        pass

if __name__ == '__main__':
    main()
