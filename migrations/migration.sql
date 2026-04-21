BEGIN;

CREATE TABLE alembic_version (
    version_num VARCHAR(32) NOT NULL, 
    CONSTRAINT alembic_version_pkc PRIMARY KEY (version_num)
);

-- Running upgrade  -> c7f66fde6909

CREATE TABLE users (
    id BIGSERIAL NOT NULL, 
    email VARCHAR NOT NULL, 
    password VARCHAR NOT NULL, 
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now(), 
    PRIMARY KEY (id), 
    UNIQUE (email)
);

INSERT INTO alembic_version (version_num) VALUES ('c7f66fde6909') RETURNING alembic_version.version_num;

COMMIT;

