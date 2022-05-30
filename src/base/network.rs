use petgraph::{Graph, Undirected};

pub struct Network {
    graph: Graph<(), (), Undirected>,
    name: String,
}

impl Network {
    /*pub fn read_hdf5(path: &Path) -> Result<Self, hdf5::Error> {
        //let h5_file = hdf5::File::open(path)?;

        //todo!()
    }*/

    pub fn line(size: u32) -> Network {
        Network {
            graph: Self::_make_grid(1, size),
            name: format!("line-{size}"),
        }
    }

    pub fn grid(size_x: u32, size_y: u32) -> Network {
        Network {
            graph: Self::_make_grid(size_x, size_y),
            name: format!("grid-{size_x}-{size_y}"),
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
