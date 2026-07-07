#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Gokoba 3D-Viewer: STEP -> interaktive HTML-Ansicht mit RAL-Waehler und Materialien.
Benoetigt: pip install cascadio  und  gltfpack (npm i -g gltfpack).
viewer-template.html muss neben dieser Datei liegen."""
import argparse, base64, datetime, os, subprocess, tempfile
try:
    import cascadio
except Exception:
    cascadio = None

_RAL_HEX = {
 "1015":"E6D2B5","1021":"F6B600","1023":"F7B500","2004":"E75B12","2009":"E15501",
 "3000":"AF2B1E","3003":"9B111E","3005":"5E2028","3020":"CC0605","4008":"924E7D",
 "5002":"20214F","5005":"1E2460","5010":"0E294B","5015":"2874B2","5017":"063971","5024":"6A93B0",
 "6005":"0F4336","6011":"6C7C59","6018":"48A43F","6024":"008351","6029":"007243",
 "7001":"8F999F","7004":"9EA0A1","7005":"6B716F","7011":"52595D","7012":"575B5A",
 "7015":"51565C","7016":"383E42","7021":"2F3234","7022":"4C4A44","7024":"45494E",
 "7030":"939183","7031":"5B6771","7035":"D7D7D7","7036":"9C9C9C","7037":"7C7F7E",
 "7040":"9DA1AA","7043":"4E5452","7046":"82898F","8001":"9D622B","8004":"8E402A",
 "8011":"5A3A29","8017":"44322D","8019":"3F3A3A","8022":"211F20","8028":"4E3B31",
 "9001":"E9E0D2","9002":"D7D5CB","9003":"ECECE7","9005":"0A0A0A","9006":"A5A5A5",
 "9007":"8F8F8C","9010":"F1EFE7","9011":"27292B","9016":"F1F1EA","9017":"2A2A2B",
}
_MODI = ["Standard (Blau)", "RAL-Farbe", "Eigene Farbe (Code)", "Verzinkt", "Edelstahl", "Grautoene (RAL 7016)"]

def _wahl_zu_srgb(modus, wert):
    m = (modus or "").lower()
    if m.startswith("standard"):
        return None
    if "grau" in m or "7016" in m:
        return [56, 62, 66]
    if "verzinkt" in m:
        return [176, 180, 183]
    if "edelstahl" in m:
        return [152, 158, 162]
    if m.startswith("ral"):
        h = _RAL_HEX.get((wert or "").strip())
        if h:
            return [int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)]
        return None
    v = (wert or "").strip().lstrip("#")
    if len(v) == 6:
        try:
            return [int(v[0:2], 16), int(v[2:4], 16), int(v[4:6], 16)]
        except ValueError:
            return None
    return None

def gokoba_dialog():
    """Kleines Fenster: fragt Stahlbau- und Gelaender-Farbe ab. Gibt sRGB-Tupel oder None (Standard) zurueck."""
    try:
        import tkinter as tk
        from tkinter import ttk
    except Exception:
        return None
    res = {"stahlbau": None, "gelaender": None, "gitter": ""}
    root = tk.Tk()
    root.title("Gokoba 3D-Viewer - Farben")
    root.resizable(False, False)
    frm = ttk.Frame(root, padding=16)
    frm.grid()
    ttk.Label(frm, text="Farben fuer die 3D-Ansicht", font=("Segoe UI", 11, "bold")).grid(column=0, row=0, columnspan=3, sticky="w", pady=(0, 12))
    m_sb, w_sb = tk.StringVar(value=_MODI[0]), tk.StringVar()
    m_ge, w_ge = tk.StringVar(value=_MODI[0]), tk.StringVar()
    def zeile(r, titel, mvar, wvar):
        ttk.Label(frm, text=titel, width=10).grid(column=0, row=r, sticky="w", pady=4)
        ttk.OptionMenu(frm, mvar, mvar.get(), *_MODI).grid(column=1, row=r, sticky="w", padx=6)
        ttk.Entry(frm, textvariable=wvar, width=16).grid(column=2, row=r, sticky="w")
    zeile(1, "Stahlbau", m_sb, w_sb)
    zeile(2, "Gelaender", m_ge, w_ge)
    _GITTER_MODI = ["Keine Textur", "Maschenweite 30x30", "Maschenweite 30x10"]
    m_gi = tk.StringVar(value=_GITTER_MODI[0])
    ttk.Label(frm, text="Gitterrost", width=10).grid(column=0, row=3, sticky="w", pady=4)
    ttk.OptionMenu(frm, m_gi, m_gi.get(), *_GITTER_MODI).grid(column=1, row=3, sticky="w", padx=6)
    ttk.Label(frm, text="Bei RAL die Nummer (z. B. 7016), bei Eigene Farbe den Code (#RRGGBB).",
              foreground="#666").grid(column=0, row=4, columnspan=3, sticky="w", pady=(10, 0))
    def ok():
        res["stahlbau"] = _wahl_zu_srgb(m_sb.get(), w_sb.get())
        res["gelaender"] = _wahl_zu_srgb(m_ge.get(), w_ge.get())
        res["gitter"] = ("3010" if "30x10" in m_gi.get() else ("3030" if "30x30" in m_gi.get() else ""))
        root.destroy()
    btns = ttk.Frame(frm)
    btns.grid(column=0, row=5, columnspan=3, pady=(16, 0), sticky="e")
    ttk.Button(btns, text="Abbrechen", command=root.destroy).grid(column=0, row=0, padx=6)
    ttk.Button(btns, text="Uebernehmen", command=ok).grid(column=1, row=0)
    root.update_idletasks()
    root.eval("tk::PlaceWindow . center")
    root.mainloop()
    return res

VORLAGE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "viewer-template.html")

def _hex(t):
    return "" if not t else "#%02X%02X%02X" % (t[0], t[1], t[2])

def erzeuge_viewer(step, out_html, model_name, def_stahl, def_gel, def_gitter, dekor_str, expiry_iso, gltfpack):
    if cascadio is None:
        raise SystemExit("Modul 'cascadio' fehlt. Bitte einmalig: pip install cascadio")
    with tempfile.TemporaryDirectory() as tmp:
        raw = os.path.join(tmp, "raw.glb")
        small = os.path.join(tmp, "small.glb")
        print("[1/3] STEP -> GLB (cascadio) ...")
        cascadio.step_to_glb(step, raw, tol_linear=0.3, tol_angular=0.22)
        print("[2/3] Komprimierung (gltfpack) ...")
        subprocess.run([gltfpack, "-i", raw, "-o", small, "-cc"], check=True)
        b64 = base64.b64encode(open(small, "rb").read()).decode()
    print("[3/3] HTML-Ansicht erzeugen ...")
    html = open(VORLAGE, encoding="utf-8").read()
    html = html.replace("__GLB_B64__", b64)
    html = html.replace("__PROJ_NAME__", model_name)
    html = html.replace("__DEF_STAHL__", def_stahl)
    html = html.replace("__DEF_GEL__", def_gel)
    html = html.replace("__GITTER_TEX__", def_gitter)
    html = html.replace("__DEKOR__", dekor_str)
    html = html.replace("__EXPIRY__", expiry_iso)
    open(out_html, "w", encoding="utf-8").write(html)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="STEP-Datei (.stp/.step)")
    ap.add_argument("--output", required=True, help="Ausgabe-HTML")
    ap.add_argument("--model-name", default="Modell", help="Projektname")
    ap.add_argument("--expiry-days", type=int, default=40, help="Gueltigkeit in Tagen")
    ap.add_argument("--gltfpack", default="gltfpack", help="Pfad zum gltfpack-Binary")
    ap.add_argument("--assets-dir", default="assets", help="(nicht mehr benoetigt, aus Kompatibilitaet akzeptiert)")
    ap.add_argument("--no-dialog", action="store_true", help="Ohne Abfrage-Dialog (Standardfarben)")
    ap.add_argument("--def-stahl", default="", help="Stahlbau-Startfarbe (verzinkt/edelstahl/grau/#RRGGBB)")
    ap.add_argument("--def-gel", default="", help="Gelaender-Startfarbe")
    ap.add_argument("--gitter", default="", help="Gitterrost-Textur: leer, 3030 oder 3010")
    ap.add_argument("--bws", default="", help="BWS-Dekor: G11013/G11213/G25013/G51013/G82313")
    ap.add_argument("--mineralit", default="", help="Mineralit-Dekor, z.B. 'granit hell'")
    ap.add_argument("--trespa1", default="", help="Trespa 1 Farbe #RRGGBB")
    ap.add_argument("--trespa2", default="", help="Trespa 2 Farbe #RRGGBB")
    ap.add_argument("--trespa3", default="", help="Trespa 3 Farbe #RRGGBB")
    ap.add_argument("--trespa4", default="", help="Trespa 4 Farbe #RRGGBB")
    args = ap.parse_args()
    def_stahl, def_gel, def_gitter = args.def_stahl, args.def_gel, args.gitter
    d_bws, d_min = args.bws, args.mineralit
    d_t = [args.trespa1, args.trespa2, args.trespa3, args.trespa4]
    # ★ Begleit-JSON neben der STEP (vom Advance-Steel-Knopf erzeugt): gleiche Basis + .json
    import json as _json
    _sc = os.path.splitext(args.input)[0] + ".json"
    if not os.path.exists(_sc):
        _sc = os.path.join(os.path.dirname(os.path.abspath(args.input)), "params.json")  # ★ vom AS-Knopf hochgeladen
    if os.path.exists(_sc):
        try:
            _j = _json.load(open(_sc, encoding="utf-8"))
            def_stahl  = _j.get("stahlbau", def_stahl) or def_stahl
            def_gel    = _j.get("gelaender", def_gel) or def_gel
            def_gitter = _j.get("gitter", def_gitter) or def_gitter
            d_bws = _j.get("bws", d_bws) or d_bws
            d_min = _j.get("mineralit", d_min) or d_min
            for _i in range(4):
                d_t[_i] = _j.get("trespa" + str(_i + 1), d_t[_i]) or d_t[_i]
        except Exception as _e:
            print("Hinweis: Begleit-JSON unlesbar:", _e)
    if not args.no_dialog:
        wahl = gokoba_dialog()
        if wahl:
            def_stahl = _hex(wahl.get("stahlbau"))
            def_gel = _hex(wahl.get("gelaender"))
            def_gitter = wahl.get("gitter") or ""
            if def_stahl: print(f"  Stahlbau-Startfarbe: {def_stahl}")
            if def_gel: print(f"  Gelaender-Startfarbe: {def_gel}")
    expiry = datetime.datetime.utcnow() + datetime.timedelta(days=args.expiry_days)
    expiry_iso = expiry.strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"Gokoba-Viewer: {os.path.basename(args.input)} | gueltig bis {expiry_iso}")
    dekor_str = "bws=" + d_bws + ";min=" + d_min + ";t1=" + d_t[0] + ";t2=" + d_t[1] + ";t3=" + d_t[2] + ";t4=" + d_t[3]  # ★ Dekor-Auswahl fuer den Viewer
    erzeuge_viewer(args.input, args.output, args.model_name, def_stahl, def_gel, def_gitter, dekor_str, expiry_iso, args.gltfpack)
    print(f"FERTIG -> {args.output}")

if __name__ == "__main__":
    main()
