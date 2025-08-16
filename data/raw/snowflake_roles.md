# Snowflake Roles
To rotate keys: use ACCOUNTADMIN, then run `ALTER USER ... SET RSA_PUBLIC_KEY`.
ETL jobs run as role `ETL_EXECUTE`.

Rotation steps:
1) Generate new keypair.
2) `ALTER USER` with new public key.
3) Update secret in scheduler.
