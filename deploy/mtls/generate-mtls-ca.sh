#!/bin/bash
# generate-mtls-ca.sh — mini внутренний CA для mTLS между Laravel-бэкендом партнёра
# (Умид, команда E-GAZ mobile) и antispoof-сервисом (egaz-02.uz).
#
# НЕ публичный CA, НЕ Let's Encrypt — это приватный корень только для этой пары
# сервисов (m2m), поэтому валидность может быть длинной, а ключевая гигиена — простой.
#
# ЗАПУСКАТЬ НА ЛОКАЛЬНОЙ ДОВЕРЕННОЙ МАШИНЕ (не на egaz-02.uz!). Приватный ключ
# корневого CA НИКОГДА не должен покидать эту машину и НЕ копируется на прод-сервер.
# На сервер уезжают только: server.crt, server.key, ca.crt (см. чек-лист для 50 CENT).
#
# Использует классическую `openssl ca` CA-БД (index.txt/serial), а не голый
# `x509 -req -CA`, специально чтобы `openssl ca -revoke` + CRL работали при
# компрометации клиентского сертификата (см. MTLS-DEPLOY-CHECKLIST.md, п. "Ротация").
#
# Usage:
#   ./generate-mtls-ca.sh init                     # один раз: создать root CA
#   ./generate-mtls-ca.sh issue-server <CN> [IP]    # серверный сертификат для nginx
#   ./generate-mtls-ca.sh issue-client <CN>         # клиентский сертификат для партнёра
#   ./generate-mtls-ca.sh revoke <cert.pem>         # отозвать сертификат + пересобрать CRL
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CA_DIR="${CA_DIR:-$SCRIPT_DIR/output}"
ROOT_CA_CN="${ROOT_CA_CN:-E-GAZ Antispoof mTLS Root CA}"
ROOT_CA_DAYS="${ROOT_CA_DAYS:-3650}"     # 10 лет — корень живёт долго, ротируется редко
LEAF_DAYS_SERVER="${LEAF_DAYS_SERVER:-825}"   # ~27 мес
LEAF_DAYS_CLIENT="${LEAF_DAYS_CLIENT:-397}"   # ~13 мес — короче, легче ротировать по расписанию
KEY_BITS_ROOT=4096
KEY_BITS_LEAF=2048

log() { echo "[mtls-ca] $*"; }

init_ca_dirs() {
    mkdir -p "$CA_DIR"/{root/private,root/certs,server,client,crl,csr}
    chmod 700 "$CA_DIR/root/private"
    [ -f "$CA_DIR/root/index.txt" ] || touch "$CA_DIR/root/index.txt"
    [ -f "$CA_DIR/root/index.txt.attr" ] || echo "unique_subject = no" > "$CA_DIR/root/index.txt.attr"
    [ -f "$CA_DIR/root/serial" ] || echo 1000 > "$CA_DIR/root/serial"
    [ -f "$CA_DIR/root/crlnumber" ] || echo 1000 > "$CA_DIR/root/crlnumber"
}

write_ca_cnf() {
    cat > "$CA_DIR/root/openssl-ca.cnf" <<EOF
[ ca ]
default_ca = CA_default

[ CA_default ]
dir               = $CA_DIR/root
database          = \$dir/index.txt
serial            = \$dir/serial
new_certs_dir     = \$dir/certs
certificate       = \$dir/ca.crt
private_key       = \$dir/private/ca.key
default_md        = sha256
default_days      = $LEAF_DAYS_SERVER
policy            = policy_loose
crlnumber         = \$dir/crlnumber
crl               = $CA_DIR/crl/ca.crl
default_crl_days  = 30
copy_extensions   = copy

[ policy_loose ]
countryName             = optional
stateOrProvinceName     = optional
organizationName        = optional
organizationalUnitName  = optional
commonName              = supplied
emailAddress            = optional

[ v3_server ]
basicConstraints = critical, CA:FALSE
keyUsage = critical, digitalSignature, keyEncipherment
extendedKeyUsage = serverAuth
subjectAltName = \$ENV::SAN

[ v3_client ]
basicConstraints = critical, CA:FALSE
keyUsage = critical, digitalSignature
extendedKeyUsage = clientAuth
EOF
}

cmd_init() {
    if [ -f "$CA_DIR/root/ca.crt" ]; then
        log "Root CA уже существует в $CA_DIR/root/ca.crt — ничего не делаю (init не перезаписывает)."
        exit 0
    fi
    init_ca_dirs
    write_ca_cnf

    log "Генерирую приватный ключ root CA (RSA ${KEY_BITS_ROOT}, с паролем)..."
    openssl genrsa -aes256 -out "$CA_DIR/root/private/ca.key" "$KEY_BITS_ROOT"
    chmod 400 "$CA_DIR/root/private/ca.key"

    log "Самоподписанный root-сертификат (${ROOT_CA_DAYS} дней, CN='${ROOT_CA_CN}')..."
    openssl req -x509 -new -key "$CA_DIR/root/private/ca.key" \
        -sha256 -days "$ROOT_CA_DAYS" \
        -subj "/CN=${ROOT_CA_CN}/O=E-GAZ/OU=antispoof-mtls" \
        -out "$CA_DIR/root/ca.crt" \
        -extensions v3_ca -config <(cat <<EOF
[req]
distinguished_name=dn
[dn]
[v3_ca]
basicConstraints = critical, CA:TRUE, pathlen:0
keyUsage = critical, keyCertSign, cRLSign
subjectKeyIdentifier = hash
EOF
)
    log "Готово: $CA_DIR/root/ca.crt (публичный, отдаём nginx и партнёру), $CA_DIR/root/private/ca.key (СЕКРЕТ, храним офлайн, НЕ копировать на сервер)."
}

cmd_issue_server() {
    local cn="${1:?Usage: issue-server <CN> [IP]}"
    local ip="${2:-}"
    [ -f "$CA_DIR/root/ca.crt" ] || { log "Сначала: ./generate-mtls-ca.sh init"; exit 1; }
    write_ca_cnf

    local san="DNS:${cn}"
    [ -n "$ip" ] && san="DNS:${cn},IP:${ip}"

    log "Генерирую серверный ключ+CSR (CN=${cn}, SAN=${san})..."
    openssl genrsa -out "$CA_DIR/server/server.key" "$KEY_BITS_LEAF"
    chmod 400 "$CA_DIR/server/server.key"
    openssl req -new -key "$CA_DIR/server/server.key" \
        -subj "/CN=${cn}/O=E-GAZ/OU=antispoof-mtls-server" \
        -out "$CA_DIR/csr/server-${cn}.csr"

    log "Подписываю серверный сертификат корневым CA (${LEAF_DAYS_SERVER} дней)..."
    SAN="$san" openssl ca -batch -config "$CA_DIR/root/openssl-ca.cnf" \
        -extensions v3_server -days "$LEAF_DAYS_SERVER" -notext -md sha256 \
        -in "$CA_DIR/csr/server-${cn}.csr" -out "$CA_DIR/server/server.crt"

    log "Готово: $CA_DIR/server/server.crt + server.key -> уезжают в nginx на egaz-02.uz."
}

cmd_issue_client() {
    local cn="${1:?Usage: issue-client <CN>, напр. egaz-laravel-partner-client-01}"
    [ -f "$CA_DIR/root/ca.crt" ] || { log "Сначала: ./generate-mtls-ca.sh init"; exit 1; }
    write_ca_cnf

    log "Генерирую клиентский ключ+CSR (CN=${cn})..."
    openssl genrsa -out "$CA_DIR/client/${cn}.key" "$KEY_BITS_LEAF"
    chmod 400 "$CA_DIR/client/${cn}.key"
    openssl req -new -key "$CA_DIR/client/${cn}.key" \
        -subj "/CN=${cn}/O=E-GAZ/OU=antispoof-mtls-client" \
        -out "$CA_DIR/csr/client-${cn}.csr"

    log "Подписываю клиентский сертификат корневым CA (${LEAF_DAYS_CLIENT} дней)..."
    SAN="" openssl ca -batch -config "$CA_DIR/root/openssl-ca.cnf" \
        -extensions v3_client -days "$LEAF_DAYS_CLIENT" -notext -md sha256 \
        -in "$CA_DIR/csr/client-${cn}.csr" -out "$CA_DIR/client/${cn}.crt"

    log "Готово: $CA_DIR/client/${cn}.crt + ${cn}.key -> передать партнёру (см. MTLS-PARTNER-HANDOFF.md), НЕ через git/чат."

    # Опционально: bundle PKCS#12 для удобства PHP/Guzzle (cURL умеет .crt/.key напрямую, но .p12 иногда удобнее)
    read -rp "[mtls-ca] Собрать также .p12-бандл для партнёра (пароль спросит openssl)? [y/N] " ans
    if [[ "$ans" =~ ^[Yy]$ ]]; then
        openssl pkcs12 -export \
            -out "$CA_DIR/client/${cn}.p12" \
            -inkey "$CA_DIR/client/${cn}.key" \
            -in "$CA_DIR/client/${cn}.crt" \
            -certfile "$CA_DIR/root/ca.crt" \
            -name "${cn}"
        log ".p12 готов: $CA_DIR/client/${cn}.p12"
    fi
}

cmd_revoke() {
    local cert="${1:?Usage: revoke <path-to-cert.pem>}"
    write_ca_cnf
    log "Отзываю сертификат $cert..."
    openssl ca -config "$CA_DIR/root/openssl-ca.cnf" -revoke "$cert"
    log "Пересобираю CRL..."
    openssl ca -config "$CA_DIR/root/openssl-ca.cnf" -gencrl -out "$CA_DIR/crl/ca.crl"
    log "CRL обновлён: $CA_DIR/crl/ca.crl — скопировать на egaz-02.uz в /etc/nginx/mtls/egaz-antispoof/ca.crl и сделать 'nginx -s reload'."
}

case "${1:-}" in
    init) cmd_init ;;
    issue-server) shift; cmd_issue_server "$@" ;;
    issue-client) shift; cmd_issue_client "$@" ;;
    revoke) shift; cmd_revoke "$@" ;;
    *) echo "Usage: $0 {init|issue-server <CN> [IP]|issue-client <CN>|revoke <cert.pem>}"; exit 1 ;;
esac
