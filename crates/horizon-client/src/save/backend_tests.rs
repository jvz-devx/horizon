use super::*;
use std::sync::{Barrier, OnceLock, atomic::AtomicUsize};

fn registry() -> &'static BlockRegistry {
    static REGISTRY: OnceLock<&'static BlockRegistry> = OnceLock::new();
    REGISTRY.get_or_init(|| {
        let facts: &[u8] = br#"{
                "defaults": {"minecraft:air": 0, "minecraft:bell": 1},
                "states": [
                    ["minecraft:air", true, false, "air", "minecraft:air"],
                    ["minecraft:bell", false, false, "bell", "minecraft:bell"]
                ]
            }"#;
        let registry = BlockRegistry::from_facts_json(facts).unwrap();
        Box::leak(Box::new(registry))
    })
}

fn biomes() -> &'static Biomes {
    static BIOMES: OnceLock<&'static Biomes> = OnceLock::new();
    BIOMES.get_or_init(|| {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("data/minecraft/worldgen/biome");
        fs::create_dir_all(&path).unwrap();
        fs::write(
            path.join("plains.json"),
            br#"{"effects":{"water_color":0}}"#,
        )
        .unwrap();
        let resources =
            horizon_assets::resource_source::ResourceRoot::directory(root.path().to_path_buf());
        Box::leak(Box::new(Biomes::load_for_server(&resources).unwrap()))
    })
}

fn tables() -> SaveTables {
    SaveTables {
        block_names: vec!["minecraft:air".into(), "minecraft:bell".into()],
        block_ids: HashMap::from([("minecraft:air".into(), 0), ("minecraft:bell".into(), 1)]),
        default_state: vec![true, true],
        block_entity: vec![false, true],
        heightmaps: crate::assets::Properties::for_tests(&[
            (true, false, false, false),
            (false, false, true, true),
        ]),
        air: 0,
        poi: vec![None, Some(("minecraft:meeting", 32))],
        component_registry: None,
        component_items: HashSet::new(),
        component_item_tags: Default::default(),
    }
}

fn assert_same_chunk(left: &Chunk, right: &Chunk, shape: LevelShape) {
    assert_eq!((left.x, left.z), (right.x, right.z));
    for y in shape.min_y..shape.min_y + shape.height {
        for z in 0..16 {
            for x in 0..16 {
                assert_eq!(left.get(x, y, z), right.get(x, y, z), "block {x} {y} {z}");
            }
        }
    }
    assert_eq!(left.structure_refs, right.structure_refs);
    match (left.saved.as_deref(), right.saved.as_deref()) {
        (Some(left), Some(right)) => {
            assert_eq!(left.last_update, right.last_update);
            assert_eq!(left.root, right.root);
            assert_eq!(left.block_entities, right.block_entities);
            assert_eq!(left.block_ticks, right.block_ticks);
            assert_eq!(left.fluid_ticks, right.fluid_ticks);
        }
        (None, None) => {}
        _ => panic!("saved chunk data differs"),
    }
}

/// Negative control: omitting the rest decode loses structure refs, block entities and ticks.
#[test]
fn rest_codec_reconstructs_the_non_section_chunk_data() {
    let registry = registry();
    let biomes = biomes();
    let tables = tables();
    let shape = LevelShape::OVERWORLD;
    let mut chunk = Chunk::empty(3, -2);
    chunk.set(1, 70, 2, 1);
    chunk
        .structure_refs
        .push(("minecraft:village".into(), (12, -8)));
    let mut extras = ChunkExtras {
        last_update: 91,
        ..ChunkExtras::default()
    };
    extras.root.insert("example:kept".into(), Tag::Long(42));
    extras.block_entities.push(Tag::compound([
        ("id", string("example:unknown_block_entity")),
        ("x", Tag::Int(49)),
        ("y", Tag::Int(70)),
        ("z", Tag::Int(-30)),
        ("custom", string("kept")),
    ]));
    extras.block_ticks.push(SavedTick {
        kind: TickTypeId(1),
        pos: BlockPos::new(49, 70, -30),
        delay: 17,
        priority: TickPriority::High,
    });
    chunk.saved = Some(Arc::new(extras));

    let encoded = encode_with(&chunk, registry, biomes, Some(&tables), shape);
    let rest = encode_rest(&chunk, registry, biomes, Some(&tables), shape);
    assert_eq!(rest.tag.get("sections"), Some(&Tag::List(Vec::new())));
    assert!(rest.tag.get("Heightmaps").is_none());
    assert!(rest.tag.get("block_ticks").is_none());
    assert!(rest.tag.get("fluid_ticks").is_none());
    assert_eq!(
        encode_rest_saved(&chunk, registry, biomes, Some(&tables), shape, None).tag,
        rest.tag
    );

    let sections = NamedTag {
        name: encoded.name.clone(),
        tag: Tag::compound([
            (
                "DataVersion",
                encoded.tag.get("DataVersion").unwrap().clone(),
            ),
            ("xPos", Tag::Int(chunk.x)),
            ("zPos", Tag::Int(chunk.z)),
            ("Status", string("minecraft:full")),
            ("sections", encoded.tag.get("sections").unwrap().clone()),
        ]),
    };
    let mut restored = decode_with(&sections, registry, biomes, Some(&tables), shape.column())
        .expect("sections decode");
    let mut rest = rest;
    // A backend stores tick lists separately and puts them back before decode_rest.
    for key in ["block_ticks", "fluid_ticks"] {
        if let Some(ticks) = encoded.tag.get(key) {
            put(&mut rest.tag, key, ticks.clone());
        }
    }
    decode_rest(&rest, &mut restored, registry, Some(&tables));
    let expected = decode_with(&encoded, registry, biomes, Some(&tables), shape.column())
        .expect("full record decodes");
    assert_same_chunk(&expected, &restored, shape);
}

#[derive(Default)]
struct MemoryBackend {
    records: Mutex<HashMap<String, HashMap<(i32, i32), NamedTag>>>,
    chunks: Mutex<HashMap<(i32, i32), Chunk>>,
    next_write_gate: Mutex<Option<Arc<Barrier>>>,
    region_chunk_queries: AtomicUsize,
}

impl MemoryBackend {
    fn seed_tag(&self, folder: &str, key: (i32, i32), tag: NamedTag) {
        self.records
            .lock()
            .unwrap()
            .entry(folder.to_owned())
            .or_default()
            .insert(key, tag);
    }
}

impl Backend for MemoryBackend {
    fn load_chunk(
        &self,
        x: i32,
        z: i32,
        _context: &BackendContext<'_>,
    ) -> Result<Option<BackendChunk>> {
        Ok(self
            .chunks
            .lock()
            .map_err(|_| "memory chunks poisoned")?
            .get(&(x, z))
            .cloned()
            .map(Box::new)
            .map(BackendChunk::Chunk))
    }

    fn read_tag(&self, folder: &str, x: i32, z: i32) -> Result<Option<NamedTag>> {
        Ok(self
            .records
            .lock()
            .map_err(|_| "memory records poisoned")?
            .get(folder)
            .and_then(|records| records.get(&(x, z)))
            .cloned())
    }

    fn write(&self, folder: &str, records: &[((i32, i32), BackendRecord<'_>)]) -> Result<()> {
        if let Some(gate) = self.next_write_gate.lock().unwrap().take() {
            gate.wait();
            gate.wait();
        }
        let mut tags = self.records.lock().map_err(|_| "memory records poisoned")?;
        let mut chunks = self.chunks.lock().map_err(|_| "memory chunks poisoned")?;
        let folder_tags = tags.entry(folder.to_owned()).or_default();
        for (key, record) in records {
            match record {
                BackendRecord::Chunk(chunk, context, blending) => {
                    folder_tags.insert(
                        *key,
                        encode_rest_saved(
                            chunk,
                            context.registry,
                            context.biomes,
                            context.tables,
                            context.shape,
                            *blending,
                        ),
                    );
                    chunks.insert(*key, (**chunk).clone());
                }
                BackendRecord::Tag(tag) => {
                    folder_tags.insert(*key, (*tag).clone());
                    if folder == "region" {
                        chunks.remove(key);
                    }
                }
                BackendRecord::Delete => {
                    folder_tags.remove(key);
                    if folder == "region" {
                        chunks.remove(key);
                    }
                }
            }
        }
        Ok(())
    }

    fn region_chunks(&self, rx: i32, rz: i32) -> Result<Vec<(i32, i32)>> {
        self.region_chunk_queries.fetch_add(1, Ordering::Relaxed);
        Ok(self
            .records
            .lock()
            .map_err(|_| "memory records poisoned")?
            .get("region")
            .into_iter()
            .flat_map(|records| records.keys().copied())
            .filter(|(x, z)| x >> 5 == rx && z >> 5 == rz)
            .collect())
    }
}

/// Negative control: bypassing `backend.write` leaves no stored records and breaks these checks.
#[test]
fn save_dir_routes_chunks_entities_and_poi_through_the_backend() {
    let root = tempfile::tempdir().unwrap();
    let backend = Arc::new(MemoryBackend::default());
    let save = SaveDir::with_backend(
        root.path().to_path_buf(),
        LevelShape::OVERWORLD,
        backend.clone(),
    );
    assert!(save.tables.set(Arc::new(tables())).is_ok());
    let old_poi = NamedTag {
        name: "".into(),
        tag: Tag::compound([
            ("DataVersion", Tag::Int(DATA_VERSION)),
            (
                "Sections",
                Tag::compound([(
                    "4",
                    Tag::compound([(
                        "Records",
                        Tag::List(vec![Tag::compound([
                            ("pos", Tag::IntArray(vec![1, 70, 2])),
                            ("type", string("minecraft:meeting")),
                            ("free_tickets", Tag::Int(7)),
                        ])]),
                    )]),
                )]),
            ),
        ]),
    };
    backend.seed_tag("poi", (0, 0), old_poi);

    let mut chunk = Chunk::empty(0, 0);
    chunk.set(1, 70, 2, 1);
    chunk
        .structure_refs
        .push(("minecraft:village".into(), (4, 5)));
    let biomes = Arc::new(biomes().clone());
    let gate = Arc::new(Barrier::new(2));
    *backend.next_write_gate.lock().unwrap() = Some(gate.clone());
    save.save_later(&[Arc::new(chunk.clone())], registry(), &biomes)
        .unwrap();

    gate.wait();
    let queued = save
        .load(0, 0, registry(), &biomes)
        .expect("queued chunk loads");
    gate.wait();
    save.flush().unwrap();
    let loaded = save
        .load(0, 0, registry(), &biomes)
        .expect("backend chunk loads");
    assert_same_chunk(&queued, &chunk, LevelShape::OVERWORLD);
    assert_same_chunk(&loaded, &chunk, LevelShape::OVERWORLD);

    save.save_entities(vec![(
        (0, 0),
        vec![Tag::compound([("id", string("example:entity"))])],
    )])
    .unwrap();
    assert!(backend.read_tag("entities", 0, 0).unwrap().is_some());
    let poi = backend.read_tag("poi", 0, 0).unwrap().expect("POI saved");
    let Some(Tag::List(records)) = poi
        .tag
        .get("Sections")
        .and_then(|s| s.get("4"))
        .and_then(|s| s.get("Records"))
    else {
        panic!("POI records are missing");
    };
    assert_eq!(number(records[0].get("free_tickets")), Some(7));
    assert!(!root.path().join("region").exists());
    assert!(!root.path().join("entities").exists());
    assert!(!root.path().join("poi").exists());
}

/// Negative control: reading a region file instead of `region_chunks` returns no old bits.
#[test]
fn old_chunk_bits_enumerates_backend_region_chunks() {
    let root = tempfile::tempdir().unwrap();
    let backend = Arc::new(MemoryBackend::default());
    backend.seed_tag(
        "region",
        (33, 64),
        NamedTag {
            name: "".into(),
            tag: Tag::compound([
                ("DataVersion", Tag::Int(3465)),
                ("xPos", Tag::Int(33)),
                ("zPos", Tag::Int(64)),
            ]),
        },
    );
    let save = SaveDir::with_backend(
        root.path().to_path_buf(),
        LevelShape::OVERWORLD,
        backend.clone(),
    );
    let bits = save.old_chunk_bits(1, 2);
    assert_eq!(bits[0] & 2, 2);
    assert_eq!(backend.region_chunk_queries.load(Ordering::Relaxed), 1);
}
