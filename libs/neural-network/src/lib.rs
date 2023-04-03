use core::panic;
use rand::Rng;
use std::iter::once;

/// Neural network
pub struct Network {
    layers: Vec<Layer>,
}

/// Number of neurons in a layer
pub type LayerTopology = usize;

struct Layer {
    neurons: Vec<Neuron>,
}

struct Neuron {
    bias: f32,
    weights: Vec<f32>,
}

impl Network {
    pub fn random(rng: &mut dyn rand::RngCore, layers: &[LayerTopology]) -> Self {
        assert!(layers.len() > 1);

        let layers = layers
            .windows(2)
            .map(|layers| Layer::random(rng, layers[0], layers[1]))
            .collect();

        Self { layers }
    }

    pub fn propagate(&self, inputs: Vec<f32>) -> Vec<f32> {
        self.layers
            .iter()
            .fold(inputs, |inputs, layer| layer.propagate(inputs))
    }

    pub fn weights(&self) -> impl Iterator<Item = f32> + '_ {
        self.layers
            .iter()
            .flat_map(|layer| layer.neurons.iter())
            .flat_map(|neuron| once(&neuron.bias).chain(&neuron.weights))
            .copied()
    }

    pub fn from_weights(layers: &[LayerTopology], weights: impl IntoIterator<Item = f32>) -> Self {
        assert!(layers.len() > 1);

        let mut weights = weights.into_iter();

        let layers = layers
            .windows(2)
            .map(|layers| Layer::from_weights(layers[0], layers[1], &mut weights))
            .collect();

        if weights.next().is_some() {
            panic!("got too many weights");
        }

        Self { layers }
    }
}

impl Layer {
    pub fn random(
        rng: &mut dyn rand::RngCore,
        input_neurons: LayerTopology,
        output_neurons: LayerTopology,
    ) -> Self {
        let neurons = (0..output_neurons)
            .map(|_| Neuron::random(rng, input_neurons))
            .collect();

        Self { neurons }
    }

    fn propagate(&self, inputs: Vec<f32>) -> Vec<f32> {
        self.neurons
            .iter()
            .map(|neuron| neuron.propagate(&inputs))
            .collect()
    }

    pub fn from_weights(
        input_size: LayerTopology,
        output_size: LayerTopology,
        weights: &mut dyn Iterator<Item = f32>,
    ) -> Self {
        let neurons = (0..output_size)
            .map(|_| Neuron::from_weights(input_size, weights))
            .collect();

        Self { neurons }
    }
}

impl Neuron {
    pub fn random(rng: &mut dyn rand::RngCore, input_size: LayerTopology) -> Self {
        let bias = rng.gen_range(-1.0..=1.0);

        let weights = (0..input_size).map(|_| rng.gen_range(-1.0..=1.0)).collect();

        Self { bias, weights }
    }

    fn propagate(&self, inputs: &[f32]) -> f32 {
        assert_eq!(inputs.len(), self.weights.len());

        let output = inputs
            .iter()
            .zip(&self.weights)
            .map(|(input, weight)| input * weight)
            .sum::<f32>();

        (self.bias + output).max(0.0)
    }

    pub fn from_weights(
        output_neurons: LayerTopology,
        weights: &mut dyn Iterator<Item = f32>,
    ) -> Self {
        let bias = weights.next().expect("got not enough weights");

        let weights = (0..output_neurons)
            .map(|_| weights.next().expect("got not enough weights"))
            .collect();

        Self { bias, weights }
    }
}

#[cfg(test)]
mod tests {
    use super::{Layer, Network, Neuron};
    use approx::assert_relative_eq;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    #[test]
    fn neuron_random() {
        let mut rng = ChaCha8Rng::from_seed(Default::default());
        let neuron = Neuron::random(&mut rng, 4);

        assert_relative_eq!(neuron.bias, -0.6255188);
        assert_relative_eq!(
            neuron.weights.as_slice(),
            [0.67383957, 0.8181262, 0.26284897, 0.5238807].as_ref()
        );
    }

    #[test]
    fn neuron_propagate() {
        let neuron = Neuron {
            bias: 0.5,
            weights: vec![-0.3, 0.8],
        };

        assert_relative_eq!(neuron.propagate(&[-10.0, -10.0]), 0.0);

        assert_relative_eq!(
            neuron.propagate(&[0.5, 1.0]),
            (-0.3 * 0.5) + (0.8 * 1.0) + 0.5
        );
    }

    #[test]
    fn layer_random() {
        let mut rng = ChaCha8Rng::from_seed(Default::default());
        let layer = Layer::random(&mut rng, 3, 2);

        assert_eq!(layer.neurons.len(), 2);

        assert_relative_eq!(layer.neurons[0].bias, -0.6255188);
        assert_relative_eq!(
            layer.neurons[0].weights.as_slice(),
            [0.67383957, 0.8181262, 0.26284897].as_ref()
        );

        assert_relative_eq!(layer.neurons[1].bias, 0.5238807);
        assert_relative_eq!(
            layer.neurons[1].weights.as_slice(),
            [-0.53516835, 0.069369674, -0.7648182].as_ref()
        );
    }

    #[test]
    fn layer_propagate() {
        let mut rng = ChaCha8Rng::from_seed(Default::default());
        let layer = Layer::random(&mut rng, 3, 2);

        let output = layer.propagate(vec![0.5, 1.0, 0.34]);
        assert_relative_eq!(output.as_slice(), [0.6188959, 0.06562802].as_ref());
    }

    #[test]
    fn network_random() {
        let mut rng = ChaCha8Rng::from_seed(Default::default());
        let network = Network::random(&mut rng, &[3, 2, 1]);

        assert_eq!(network.layers.len(), 2);
        assert_eq!(network.layers[0].neurons.len(), 2);
        assert_eq!(network.layers[1].neurons.len(), 1);

        assert_relative_eq!(network.layers[0].neurons[0].bias, -0.6255188);
        assert_relative_eq!(
            network.layers[0].neurons[0].weights.as_slice(),
            [0.67383957, 0.8181262, 0.26284897].as_ref()
        );

        assert_relative_eq!(network.layers[0].neurons[1].bias, 0.5238807);
        assert_relative_eq!(
            network.layers[0].neurons[1].weights.as_slice(),
            [-0.53516835, 0.069369674, -0.7648182].as_ref()
        );

        assert_relative_eq!(network.layers[1].neurons[0].bias, -0.102499366);
        assert_relative_eq!(
            network.layers[1].neurons[0].weights.as_slice(),
            [-0.48879617, -0.19277132].as_ref()
        );
    }

    #[test]
    fn network_propagate() {
        let mut rng = ChaCha8Rng::from_seed(Default::default());
        let network = Network::random(&mut rng, &[3, 3, 1]);

        let output = network.propagate(vec![0.0, 0.0, 0.0]);
        assert_relative_eq!(output.as_slice(), [0.508711].as_ref());
    }
}
