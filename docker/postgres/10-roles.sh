#!/bin/sh
# Database roles (D-026). Postgres runs this ONCE, on the first start of an empty
# data volume (docker-entrypoint-initdb.d). Passwords come from the environment.
#
#   scorecard_owner  runs migrations: may create and alter tables
#   scorecard_app    the API: may only SELECT and INSERT. No UPDATE, no DELETE,
#                    so the audit log is append-only in the database itself (D-025).
set -eu
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" \
     -v owner_pw="$SCORECARD_OWNER_PASSWORD" -v app_pw="$SCORECARD_APP_PASSWORD" \
     -v db="$POSTGRES_DB" <<'SQL'
CREATE ROLE scorecard_owner LOGIN PASSWORD :'owner_pw';
CREATE ROLE scorecard_app   LOGIN PASSWORD :'app_pw';

REVOKE ALL ON DATABASE :"db" FROM PUBLIC;
GRANT CONNECT ON DATABASE :"db" TO scorecard_owner, scorecard_app;

REVOKE ALL ON SCHEMA public FROM PUBLIC;
GRANT USAGE, CREATE ON SCHEMA public TO scorecard_owner;
GRANT USAGE ON SCHEMA public TO scorecard_app;

-- Applies to every table the owner creates from now on, i.e. through migrations.
ALTER DEFAULT PRIVILEGES FOR ROLE scorecard_owner IN SCHEMA public
    GRANT SELECT, INSERT ON TABLES TO scorecard_app;
SQL
