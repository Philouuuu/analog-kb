-- ============================================================
-- analog-kb — Supabase schema
-- Colle ce SQL dans : Supabase → SQL Editor → New query → Run
-- ============================================================

-- 1. Extension pgvector
create extension if not exists vector;

-- 2. Table documents (métadonnées)
create table if not exists documents (
  id          uuid        primary key default gen_random_uuid(),
  title       text        not null,
  source      text        not null check (source in ('drive', 'obsidian', 'github')),
  category    text,
  file_path   text        unique,
  file_type   text,
  created_at  timestamptz default now()
);

-- 3. Table chunks (contenu + vecteurs)
--    Gemini text-embedding-004 → 768 dimensions
create table if not exists chunks (
  id            uuid    primary key default gen_random_uuid(),
  document_id   uuid    not null references documents(id) on delete cascade,
  content       text    not null,
  embedding     vector(768),
  page_num      integer,
  chunk_index   integer,
  created_at    timestamptz default now()
);

-- 4. Index ANN pour la recherche vectorielle rapide
create index if not exists chunks_embedding_idx
  on chunks using ivfflat (embedding vector_cosine_ops)
  with (lists = 100);

-- 5. RLS — la clé anon peut lire, seule la service key peut écrire
alter table documents enable row level security;
alter table chunks    enable row level security;

create policy "anon_read_documents" on documents
  for select using (true);

create policy "anon_read_chunks" on chunks
  for select using (true);

-- 6. Fonction de recherche vectorielle
create or replace function search_chunks(
  query_embedding vector(768),
  match_count     int     default 6,
  filter_category text    default null
)
returns table (
  chunk_id    uuid,
  document_id uuid,
  title       text,
  category    text,
  source      text,
  content     text,
  page_num    integer,
  similarity  float
)
language plpgsql
as $$
begin
  return query
  select
    c.id,
    c.document_id,
    d.title,
    d.category,
    d.source,
    c.content,
    c.page_num,
    1 - (c.embedding <=> query_embedding) as similarity
  from chunks c
  join documents d on c.document_id = d.id
  where
    filter_category is null
    or d.category = filter_category
  order by c.embedding <=> query_embedding
  limit match_count;
end;
$$;
