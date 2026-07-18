# Чек-лист раскатки mTLS перед antispoof — для 50 CENT

Готовит: BUSTA RHYMES (транспортный слой). Не выполнял деплой сам — нет
подтверждённого SSH-доступа к `egaz-02.uz` из этой сессии (hostname не резолвится
с рабочей машины). Всё ниже — конфиги/скрипты, готовые к ручной раскатке.

**Не трогать:** `app/main.py`, `app/config.py`, логику IP-allowlist, anti-replay —
это SNOOP/KENDRICK. Моя зона — nginx, TLS/mTLS-сертификаты, bind uvicorn, firewall.

## 0. Разведка ДО изменений (обязательно, сохранить вывод)

```bash
ss -tlnp | grep 8090          # подтвердить, что uvicorn сейчас слушает 0.0.0.0:8090
sudo iptables -L -n -v | head -50
which nginx || echo "nginx НЕ установлен"
systemctl status antispoof.service
curl -sk https://127.0.0.1:8090/health   # текущее поведение без прокси
```

⚠️ Если сервис слушает **не только** `0.0.0.0:8090` (или уже есть внешние клиенты,
активно бьющие в 8090 напрямую) — сообщить владельцу ДО переключения на loopback,
это точка, где можно оборвать доступ партнёру.

## 1. Генерация CA/сертификатов — НЕ на сервере

Скрипт `generate-mtls-ca.sh` запускать на локальной доверенной машине (не на
egaz-02.uz). Приватный ключ root CA никогда не должен попасть на прод-сервер.

```bash
cd deploy/mtls
./generate-mtls-ca.sh init
./generate-mtls-ca.sh issue-server egaz-02.uz <реальный_IP_egaz-02.uz>
./generate-mtls-ca.sh issue-client egaz-laravel-partner-client-01
```

Результат в `deploy/mtls/output/`:
- `root/ca.crt` — публичный корень (нужен и nginx, и партнёру)
- `root/private/ca.key` — СЕКРЕТ, остаётся офлайн у владельца (password-manager/шифрованный диск), НЕ копировать никуда
- `server/server.crt` + `server/server.key` — на сервер, в nginx
- `client/egaz-laravel-partner-client-01.crt` + `.key` (+ опционально `.p12`) — партнёру, см. `MTLS-PARTNER-HANDOFF.md`

## 2. Установка nginx (если отсутствует)

```bash
sudo apt update && sudo apt install -y nginx
sudo systemctl enable nginx
```

## 3. Разложить сертификаты на сервере

```bash
sudo mkdir -p /etc/nginx/mtls/egaz-antispoof
sudo install -m 600 -o root -g root server.key  /etc/nginx/mtls/egaz-antispoof/server.key
sudo install -m 644 -o root -g root server.crt  /etc/nginx/mtls/egaz-antispoof/server.crt
sudo install -m 644 -o root -g root ca.crt      /etc/nginx/mtls/egaz-antispoof/ca.crt
```

## 4. Конфиг nginx

```bash
sudo cp deploy/mtls/nginx-antispoof-mtls.conf /etc/nginx/sites-available/antispoof-mtls.conf
sudo ln -s /etc/nginx/sites-available/antispoof-mtls.conf /etc/nginx/sites-enabled/antispoof-mtls.conf
sudo nginx -t                      # ОБЯЗАТЕЛЬНО перед reload
sudo systemctl reload nginx
```

## 5. Проверка mTLS ДО переключения bind uvicorn

В этот момент uvicorn ещё слушает `0.0.0.0:8090` — старый прямой доступ ещё
жив, это наша страховка на случай проблем с nginx.

```bash
# Без клиентского сертификата — ожидаем отказ TLS handshake (400/495/reset)
curl -sk https://egaz-02.uz/health

# С клиентским сертификатом партнёра — ожидаем 200
curl -sk --cert client/egaz-laravel-partner-client-01.crt \
         --key  client/egaz-laravel-partner-client-01.key \
         --cacert root/ca.crt \
         https://egaz-02.uz/health
```

Если второй curl не проходит — НЕ переключать bind uvicorn (шаг 6), чинить nginx.

## 6. Переключить uvicorn на loopback (необратимо для прямого доступа — ПРЕДУПРЕДИТЬ владельца/Умида заранее о времени окна)

```bash
sudo cp deploy/mtls/antispoof-loopback.service /etc/systemd/system/antispoof.service
sudo systemctl daemon-reload
sudo systemctl restart antispoof.service
ss -tlnp | grep 8090     # ожидаем 127.0.0.1:8090, НЕ 0.0.0.0:8090
```

⚠️ С этого момента партнёрский Laravel должен ходить ТОЛЬКО через
`https://egaz-02.uz/` с mTLS-клиентским сертификатом. Прямой HTTP на `:8090`
снаружи больше недоступен. Это и есть переключатель `RzaLivenessGate`, которого
ждёт прод-страж Умида ("https до RZA").

## 7. Firewall — закрыть 8090 снаружи явно (defense-in-depth)

Даже после loopback-bind — закрыть порт правилом firewall на случай, если
кто-то по ошибке вернёт `--host 0.0.0.0`:

```bash
sudo ufw allow 443/tcp
sudo ufw deny 8090/tcp
# либо iptables, если ufw не используется:
# sudo iptables -A INPUT -p tcp --dport 8090 -j DROP
```

**Перед применением — проверить, что правило не режет SSH (порт 22) и что
8090 не нужен ещё кому-то на этом сервере.**

## 8. Smoke-тест после переключения

```bash
curl -sk --cert .../client.crt --key .../client.key --cacert .../ca.crt \
     -H "X-Service-Token: <текущий_секрет>" \
     -X POST https://egaz-02.uz/verify -F "file=@face.jpg"
```

Ожидаем штатный ответ сервиса (200/структурированный JSON), не 401/403/TLS-ошибку.

## 9. Что сообщить владельцу/Умиду после раскатки

- URL для Laravel: `https://egaz-02.uz/` (порт 443, был 8090 напрямую — меняется схема доступа)
- Дата смены — нужна синхронизация окна отключения старого прямого HTTP
- Файлы для партнёра — см. `MTLS-PARTNER-HANDOFF.md`

## Известная находка, не в моей зоне, но критична — сообщить SNOOP/KENDRICK/владельцу

`app/main.py` строит IP-allowlist по `request.client.host`, и `127.0.0.0/8`
**уже входит** в `ALLOWED_NETWORKS`. После введения nginx-прокси на `127.0.0.1:8090`
приложение видит peer-адрес `127.0.0.1` для ЛЮБОГО внешнего запроса, прошедшего
через nginx — app-level IP-allowlist фактически перестаёт что-либо фильтровать
(тихо "открывается" для всех, кто достучался до nginx). mTLS + nginx-allow/deny
(см. закомментированный блок в конфиге) компенсируют это на транспортном уровне,
но правильный долгосрочный фикс — на стороне приложения (например, доверять
`X-Forwarded-For`/`X-Real-IP` только от nginx через `proxy_set_header`, что уже
проброшено в конфиге, и делать allowlist по нему). Это код — не трогаю сам,
эскалирую.
