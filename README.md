# Gokoba 3D-Viewer — Einrichtung (Weg 2: GitHub-Cloud)

Dieses Paket macht aus einer STEP-Datei automatisch einen Online-3D-Viewer.
Ablauf später im Betrieb: In Advance Steel auf den **3D-Viewer**-Button klicken →
warten (ca. 3–5 Min) → fertigen Link an den Kunden schicken. Der Link ist **20 Tage** gültig.

Die Einrichtung machst du **einmal**. Danach läuft alles per Knopfdruck.

---

## Was passiert technisch?

1. Der Button lädt die (komprimierte) STEP-Datei zu deinem GitHub-Repository hoch.
2. GitHub führt automatisch die Konvertierung durch (kostenlos, in der Cloud).
3. Der fertige Viewer landet auf GitHub Pages unter einem zufälligen Link.
4. Ein täglicher Aufräum-Job löscht Viewer, die älter als 20 Tage sind.

Alles kostenlos. Du brauchst nichts auf deinem PC zu installieren.

---

## Schritt 1 — GitHub-Konto

Falls noch keins vorhanden: auf https://github.com kostenlos registrieren.
Merke dir deinen **Benutzernamen** (z. B. `paulthomas`).

## Schritt 2 — Repository anlegen

1. Oben rechts auf **+** → **New repository**.
2. **Repository name:** `gokoba-viewer`
3. Sichtbarkeit: **Public** (nötig, damit GitHub Pages kostenlos funktioniert –
   die Inhalte sind nur über die zufälligen, nicht erratbaren Links erreichbar).
4. **Create repository**.

## Schritt 3 — Dateien hochladen

1. Im neuen, leeren Repository: **uploading an existing file** anklicken
   (oder **Add file → Upload files**).
2. Den **gesamten Inhalt** dieses Pakets hochladen (am einfachsten: alle Dateien
   und Ordner markieren und ins Browserfenster ziehen). Wichtig sind:
   - `convert.py`
   - der Ordner `.github/` (mit den Workflows)
   - der Ordner `assets/`
   - der Ordner `docs/`
   - der Ordner `jobs/`
3. Unten auf **Commit changes**.

> Hinweis: Falls GitHub den Ordner `.github` beim Ziehen ausblendet, lade die Datei
> `.github/workflows/convert.yml` und `.github/workflows/cleanup.yml` notfalls einzeln
> über **Add file → Create new file** an und füge den Inhalt ein
> (Dateiname inkl. Pfad eingeben: `.github/workflows/convert.yml`).

## Schritt 4 — GitHub Pages aktivieren

1. Im Repository oben auf **Settings**.
2. Links auf **Pages**.
3. Unter **Build and deployment → Source:** „Deploy from a branch".
4. **Branch:** `main`, Ordner: **/docs**. → **Save**.

Damit ist deine Viewer-Adresse:
`https://DEIN_GITHUB_NAME.github.io/gokoba-viewer/`

## Schritt 5 — Zugangs-Token erstellen

Der Button braucht einen Schlüssel, um Dateien hochladen zu dürfen.

1. Rechts oben auf dein Profilbild → **Settings**.
2. Ganz unten links: **Developer settings**.
3. **Personal access tokens → Tokens (classic) → Generate new token (classic)**.
4. **Note:** `Gokoba Plugin`. **Expiration:** z. B. „No expiration" oder 1 Jahr.
5. Häkchen setzen bei **`repo`** (gibt Schreibrechte auf deine Repos).
6. **Generate token**. Den angezeigten Token (beginnt mit `ghp_…`) **sofort kopieren**
   – er wird nur einmal angezeigt.

## Schritt 6 — Zugang ins Plugin eintragen

In `Class1.cs`, in der Klasse `GokobaViewerUploader`, ganz oben die vier Werte setzen:

```csharp
private const string GITHUB_USER   = "paulthomas";        // dein GitHub-Name
private const string GITHUB_REPO   = "gokoba-viewer";     // so wie in Schritt 2
private const string GITHUB_BRANCH = "main";
private const string GITHUB_TOKEN  = "ghp_xxxxxxxxxxxxx"; // der Token aus Schritt 5
```

## Schritt 7 — Projekt-Verweise prüfen

Das Plugin braucht diese Verweise (Visual Studio → Projekt → **Verweise** →
**Verweis hinzufügen** → **Assemblys → Framework**):

- **System.Net.Http**
- **System.IO.Compression**
- **System.Windows.Forms** (meist schon vorhanden)

Dann das Plugin neu kompilieren.

---

## Fertig — so nutzt du es

1. Modell in Advance Steel öffnen und **speichern**.
2. Mit dem **STEP**-Button eine STEP-Datei erzeugen
   (sie muss gleich heißen wie das Modell und im selben Ordner liegen).
3. Auf den **3D-Viewer**-Button klicken.
4. In der Befehlszeile erscheint der Link (auch in der Zwischenablage).
5. Nach ein paar Minuten ist der Viewer unter dem Link erreichbar – an den Kunden schicken.

Der Link funktioniert 20 Tage, danach zeigt er automatisch „Link abgelaufen".

---

## Wenn etwas hakt

- **„GitHub-Antwort 401/403"** → Token falsch oder ohne `repo`-Recht. Token neu erstellen.
- **„GitHub-Antwort 404"** → `GITHUB_USER` oder `GITHUB_REPO` stimmt nicht.
- **Link zeigt 404, auch nach 5 Min** → unter **Actions** im Repo schauen, ob der
  Lauf „STEP zu 3D-Viewer" grün ist. Beim allerersten Mal Pages einmal kurz Zeit geben.
- **Konvertierung schlägt fehl** → unter **Actions** den roten Lauf öffnen, die
  Fehlermeldung kopieren und mir schicken.
