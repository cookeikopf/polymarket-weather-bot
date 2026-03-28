#!/bin/bash
# ═══════════════════════════════════════════════════════════════════
# Polymarket Weather Bot — Automatisches VPS-Setup
# ═══════════════════════════════════════════════════════════════════
# Kopiere dieses gesamte Script in dein Terminal und drücke Enter.
# Es installiert ALLES automatisch und startet den Bot.
# ═══════════════════════════════════════════════════════════════════

set -e
echo "╔══════════════════════════════════════════════╗"
echo "║  Polymarket Weather Bot — Setup startet...   ║"
echo "╚══════════════════════════════════════════════╝"

# 1) System updaten
echo ""
echo "▶ [1/8] System wird aktualisiert..."
apt update -y && apt upgrade -y

# 2) Python + Git + tmux installieren
echo ""
echo "▶ [2/8] Python, Git und tmux werden installiert..."
apt install -y python3 python3-pip python3-venv git tmux curl

# 3) Geoblock-Check
echo ""
echo "▶ [3/8] Prüfe ob Polymarket von hier erreichbar ist..."
GEO=$(curl -s https://ipinfo.io/country)
echo "  Dein VPS-Standort: $GEO"
if [[ "$GEO" == "US" || "$GEO" == "CN" || "$GEO" == "FR" || "$GEO" == "DE" || "$GEO" == "RU" ]]; then
    echo "  ⚠️  WARNUNG: Dieses Land könnte geblockt sein!"
else
    echo "  ✅ Land ist nicht auf der Polymarket-Blockliste"
fi

# 4) Bot von GitHub klonen
echo ""
echo "▶ [4/8] Bot wird von GitHub heruntergeladen..."
cd /root
if [ -d "polymarket-weather-bot" ]; then
    echo "  Bot-Ordner existiert bereits, wird aktualisiert..."
    cd polymarket-weather-bot
    git pull origin main
else
    git clone https://github.com/cookeikopf/polymarket-weather-bot.git
    cd polymarket-weather-bot
fi

# 5) Python Virtual Environment + Dependencies
echo ""
echo "▶ [5/8] Python-Abhängigkeiten werden installiert..."
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# 6) .env Datei erstellen
echo ""
echo "▶ [6/8] Konfiguration wird erstellt..."
cat > .env << 'ENVFILE'
POLYMARKET_PRIVATE_KEY=0xd1a3f6b9738889dd582db4c75acef3ac65f7621e89f3d4d594c8a8e230690924
POLYMARKET_FUNDER_ADDRESS=0x306a227A17AdAb2ae5D42e74afd41F0756D108FD
POLYMARKET_SIGNATURE_TYPE=2
POLYMARKET_BANKROLL=100
POLY_BUILDER_API_KEY=019cfc94-f8f4-7d98-8a9f-74ab079858db
POLY_BUILDER_SECRET=syTJGzJj8qR6qYQ47lHPSM8gssXB-uWygeivoc_yEJA=
POLY_BUILDER_PASSPHRASE=5e48a7ddaee9792519ac22333fa82c484532ba5506e8f36d42d0fcfad4993c15
OPEN_METEO_API_KEY=wjrcKzLOeLkcCnzx
ORDER_STRATEGY=adaptive
ENVFILE
echo "  ✅ .env erstellt"

# 7) Bot-Status testen
echo ""
echo "▶ [7/8] Bot wird getestet..."
source venv/bin/activate
python3 main.py status 2>&1 || echo "  (Status-Check abgeschlossen)"

# 8) Bot in tmux starten (läuft im Hintergrund weiter)
echo ""
echo "▶ [8/8] Bot wird im Hintergrund gestartet..."

# Erstelle Start-Script
cat > /root/start_bot.sh << 'STARTSCRIPT'
#!/bin/bash
cd /root/polymarket-weather-bot
source venv/bin/activate
python3 main.py scan
STARTSCRIPT
chmod +x /root/start_bot.sh

# Erstelle Live-Start-Script
cat > /root/start_bot_live.sh << 'LIVESCRIPT'
#!/bin/bash
cd /root/polymarket-weather-bot
source venv/bin/activate
echo "═══════════════════════════════════════"
echo "  LIVE TRADING GESTARTET"
echo "  Drücke Ctrl+C zum Stoppen"
echo "═══════════════════════════════════════"
python3 main.py live 1440
LIVESCRIPT
chmod +x /root/start_bot_live.sh

# Erstelle Scan-Script (Paper Trading)
cat > /root/start_bot_paper.sh << 'PAPERSCRIPT'
#!/bin/bash
cd /root/polymarket-weather-bot
source venv/bin/activate
echo "═══════════════════════════════════════"
echo "  PAPER TRADING (kein echtes Geld)"
echo "  Drücke Ctrl+C zum Stoppen"
echo "═══════════════════════════════════════"
python3 main.py paper 1440
PAPERSCRIPT
chmod +x /root/start_bot_paper.sh

# Auto-Start bei Server-Neustart
cat > /etc/systemd/system/polybot.service << 'SYSTEMD'
[Unit]
Description=Polymarket Weather Bot
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/root/polymarket-weather-bot
ExecStart=/root/polymarket-weather-bot/venv/bin/python3 main.py live 1440
Restart=always
RestartSec=60
Environment=PATH=/root/polymarket-weather-bot/venv/bin:/usr/bin:/bin

[Install]
WantedBy=multi-user.target
SYSTEMD

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║                                                          ║"
echo "║   ✅ SETUP ABGESCHLOSSEN!                               ║"
echo "║                                                          ║"
echo "║   Befehle die du kennen musst:                           ║"
echo "║                                                          ║"
echo "║   Paper Trading starten (EMPFOHLEN als erstes):          ║"
echo "║     tmux new -s bot '/root/start_bot_paper.sh'           ║"
echo "║                                                          ║"
echo "║   Live Trading starten:                                  ║"
echo "║     tmux new -s bot '/root/start_bot_live.sh'            ║"
echo "║                                                          ║"
echo "║   Bot im Hintergrund anschauen:                          ║"
echo "║     tmux attach -t bot                                   ║"
echo "║                                                          ║"
echo "║   Bot-Fenster verlassen (läuft weiter):                  ║"
echo "║     Ctrl+B, dann D                                       ║"
echo "║                                                          ║"
echo "║   Bot stoppen:                                            ║"
echo "║     tmux kill-session -t bot                             ║"
echo "║                                                          ║"
echo "║   Auto-Start aktivieren (Bot startet nach Reboot):      ║"
echo "║     systemctl enable polybot                             ║"
echo "║     systemctl start polybot                              ║"
echo "║                                                          ║"
echo "║   Auto-Start Status prüfen:                              ║"
echo "║     systemctl status polybot                             ║"
echo "║                                                          ║"
echo "║   Bot-Logs anschauen:                                    ║"
echo "║     journalctl -u polybot -f                             ║"
echo "║                                                          ║"
echo "║   Bankroll ändern:                                        ║"
echo "║     nano /root/polymarket-weather-bot/.env                ║"
echo "║     (POLYMARKET_BANKROLL=200 z.B.)                       ║"
echo "║                                                          ║"
echo "╚══════════════════════════════════════════════════════════╝"
