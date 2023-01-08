use petgraph::{Graph, Undirected};
use rand::distributions::{Alphanumeric, Bernoulli};
use rand::Rng;
use serde::Serialize;
use serde_json::json;
use std::collections::HashMap;
use std::fs;
use std::path::Path;

pub struct Network {
    graph: Graph<(), (), Undirected>,
    name: String,
    conf: serde_json::Value,
}

#[derive(Serialize)]
pub struct NetworkDescription<'a> {
    pub name: &'a str,
    pub nodes: usize,
    pub edges: usize,
    #[serde(flatten)]
    pub conf: &'a serde_json::Value,
}

impl Network {
    /*pub fn read_hdf5(path: &Path) -> Result<Self, hdf5::Error> {
        //let h5_file = hdf5::File::open(path)?;

        //todo!()
    }*/

    pub fn load_json(path: &Path) -> Network {
        let data: Vec<[u32; 2]> = serde_json::from_str(&fs::read_to_string(path).unwrap()).unwrap();
        let mut nodes: HashMap<u32, _> = HashMap::new();
        let mut graph = Graph::new_undirected();
        for [n1, n2] in data {
            let node1 = *nodes.entry(n1).or_insert_with(|| graph.add_node(()));
            let node2 = *nodes.entry(n2).or_insert_with(|| graph.add_node(()));
            graph.add_edge(node1, node2, ());
        }
        Network {
            graph,
            name: path.to_string_lossy().to_string(),
            conf: serde_json::Value::Null,
        }
    }

    pub fn line(size: u32) -> Network {
        Network {
            graph: Self::_make_grid(1, size),
            name: "line".to_string(),
            conf: serde_json::Value::Null,
        }
    }

    pub fn grid(size_x: u32, size_y: u32) -> Network {
        Network {
            graph: Self::_make_grid(size_x, size_y),
            name: "grid".to_string(),
            conf: json!({ "x": size_x, "y": size_y }),
        }
    }

    pub fn random(rng: &mut impl Rng, n_nodes: u32, prob: f64) -> Network {
        let mut graph = Graph::new_undirected();
        let nodes: Vec<_> = (0..n_nodes).map(|_| graph.add_node(())).collect();
        let dist = Bernoulli::new(prob).unwrap();
        for (i, n) in nodes.iter().enumerate() {
            for m in &nodes[i + 1..] {
                if rng.sample(dist) {
                    graph.add_edge(*n, *m, ());
                }
            }
        }
        let uid: String = rng
            .sample_iter(&Alphanumeric)
            .take(7)
            .map(char::from)
            .collect();

        Network {
            graph,
            name: "rnd".to_string(),
            conf: json!({"p": prob, "uid": uid }),
        }
    }

    fn _make_grid(size_x: u32, size_y: u32) -> Graph<(), (), Undirected> {
        let mut graph = Graph::new_undirected();

        if size_x > 0 && size_y > 0 {
            let mut buffer = Vec::with_capacity(size_x as usize);
            buffer.push(graph.add_node(()));
            for i in 1..size_x as usize {
                let node = graph.add_node(());
                graph.add_edge(buffer[i - 1], node, ());
                buffer.push(node);
            }
            for _ in 1..size_y {
                let node = graph.add_node(());
                graph.add_edge(buffer[0], node, ());
                buffer[0] = node;
                for i in 1..size_x as usize {
                    let node = graph.add_node(());
                    graph.add_edge(buffer[i], node, ());
                    graph.add_edge(buffer[i - 1], node, ());
                    buffer[i] = node;
                }
            }
        }
        graph
    }

    pub fn description(&self) -> NetworkDescription {
        NetworkDescription {
            name: &self.name,
            nodes: self.graph.node_count(),
            edges: self.graph.edge_count(),
            conf: &self.conf,
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn graph(&self) -> &Graph<(), (), Undirected> {
        &self.graph
    }

    pub fn node_count(&self) -> usize {
        self.graph.node_count()
    }

    pub fn edge_count(&self) -> usize {
        self.graph.edge_count()
    }
}

#[cfg(test)]
mod test {
    use super::Network;

    #[test]
    fn test_line() {
        let net = Network::line(4);
        assert_eq!(net.node_count(), 4);
        assert_eq!(net.edge_count(), 3);
    }

    #[test]
    fn test_grid() {
        let net = Network::grid(3, 4);
        assert_eq!(net.node_count(), 12);
        assert_eq!(net.edge_count(), 17);

        assert_eq!(net.graph.edges(0.into()).count(), 2);
        assert_eq!(net.graph.edges(1.into()).count(), 3);
        assert_eq!(net.graph.edges(2.into()).count(), 2);
        assert_eq!(net.graph.edges(3.into()).count(), 3);
        assert_eq!(net.graph.edges(4.into()).count(), 4);
        assert_eq!(net.graph.edges(10.into()).count(), 3);
        assert_eq!(net.graph.edges(11.into()).count(), 2);
    }
}
