use rand::seq::SliceRandom;
use rand::Rng;
use rand::RngCore;
use std::ops::Index;

//////////////////////////////////
// Traits
//////////////////////////////////
pub trait Individual {
    fn create(chromosome: Chromosome) -> Self;

    fn chromosome(&self) -> &Chromosome;

    fn fitness(&self) -> f32;
}

pub trait SelectionMethod {
    fn select<'a, I>(&self, rng: &mut dyn RngCore, population: &'a [I]) -> &'a I
    where
        I: Individual;
}

pub trait CrossoverMethod {
    fn crossover(
        &self,
        rng: &mut dyn RngCore,
        parent_a: &Chromosome,
        parent_b: &Chromosome,
    ) -> Chromosome;
}

pub trait MutationMethod {
    fn mutate(&self, rng: &mut dyn RngCore, child: &mut Chromosome);
}

//////////////////////////////////
// Genetic Algorithm
//////////////////////////////////
pub struct Statistics {
    min_fitness: f32,
    max_fitness: f32,
    avg_fitness: f32,
}

pub struct GeneticAlgorithm<S, C, M> {
    selection_method: S,
    crossover_method: C,
    mutation_method: M,
}

impl<S, C, M> GeneticAlgorithm<S, C, M>
where
    S: SelectionMethod,
    C: CrossoverMethod,
    M: MutationMethod,
{
    pub fn new(selection_method: S, crossover_method: C, mutation_method: M) -> Self {
        Self {
            selection_method,
            crossover_method,
            mutation_method,
        }
    }

    pub fn evolve<I>(&self, rng: &mut dyn RngCore, population: &[I]) -> (Vec<I>, Statistics)
    where
        I: Individual,
    {
        assert!(!population.is_empty());

        let new_population = population
            .iter()
            .map(|_| {
                let parent_a = self.selection_method.select(rng, population).chromosome();
                let parent_b = self.selection_method.select(rng, population).chromosome();

                let mut child = self.crossover_method.crossover(rng, parent_a, parent_b);

                self.mutation_method.mutate(rng, &mut child);

                I::create(child)
            })
            .collect();

        let stats = Statistics::new(population);

        (new_population, stats)
    }
}

impl Statistics {
    fn new<I>(population: &[I]) -> Self
    where
        I: Individual,
    {
        assert!(!population.is_empty());

        let mut min_fitness = population[0].fitness();
        let mut max_fitness = min_fitness;
        let mut sum_fitness = 0.0;

        for individual in population {
            let fitness = individual.fitness();

            min_fitness = min_fitness.min(fitness);
            max_fitness = max_fitness.max(fitness);
            sum_fitness += fitness;
        }

        Self {
            min_fitness,
            max_fitness,
            avg_fitness: sum_fitness / (population.len() as f32),
        }
    }

    pub fn min_fitness(&self) -> f32 {
        self.min_fitness
    }

    pub fn max_fitness(&self) -> f32 {
        self.max_fitness
    }

    pub fn avg_fitness(&self) -> f32 {
        self.avg_fitness
    }
}

impl Default for Statistics {
    fn default() -> Self {
        Self {
            min_fitness: 0.0,
            max_fitness: 0.0,
            avg_fitness: 0.0,
        }
    }
}

//////////////////////////////////
// Roulette Wheel Selection
//////////////////////////////////
pub struct RouletteWheelSelection;

impl RouletteWheelSelection {
    pub fn new() -> Self {
        Self
    }
}

impl Default for RouletteWheelSelection {
    fn default() -> Self {
        Self::new()
    }
}

impl SelectionMethod for RouletteWheelSelection {
    fn select<'a, I>(&self, rng: &mut dyn RngCore, population: &'a [I]) -> &'a I
    where
        I: Individual,
    {
        population
            .choose_weighted(rng, |individual| individual.fitness())
            .expect("got empty population")
    }
}

//////////////////////////////////
// Uniform Crossover
//////////////////////////////////
#[derive(Clone, Debug)]
pub struct UniformCrossover;

impl UniformCrossover {
    pub fn new() -> Self {
        Self
    }
}

impl Default for UniformCrossover {
    fn default() -> Self {
        Self::new()
    }
}

impl CrossoverMethod for UniformCrossover {
    fn crossover(
        &self,
        rng: &mut dyn RngCore,
        parent_a: &Chromosome,
        parent_b: &Chromosome,
    ) -> Chromosome {
        assert_eq!(parent_a.len(), parent_b.len());

        let parent_a = parent_a.iter();
        let parent_b = parent_b.iter();

        parent_a
            .zip(parent_b)
            .map(|(&a, &b)| if rng.gen_bool(0.5) { a } else { b })
            .collect()
    }
}

//////////////////////////////////
// Gaussian Mutation
//////////////////////////////////
#[derive(Clone, Debug)]
pub struct GaussianMutation {
    /// Probability of changing a gene:
    /// - 0.0 = no genes will be touched
    /// - 1.0 = all genes will be touched
    chance: f32,

    /// Magnitude of that change:
    /// - 0.0 = touched genes will not be modified
    /// - 3.0 = touched genes will be += or -= by at most 3.0
    coeff: f32,
}

impl GaussianMutation {
    pub fn new(chance: f32, coeff: f32) -> Self {
        assert!((0.0..=1.0).contains(&chance));

        Self { chance, coeff }
    }
}

impl MutationMethod for GaussianMutation {
    fn mutate(&self, rng: &mut dyn RngCore, child: &mut Chromosome) {
        for gene in child.iter_mut() {
            let sign = if rng.gen_bool(0.5) { -1.0 } else { 1.0 };

            if rng.gen_bool(self.chance as _) {
                *gene += sign * self.coeff * rng.gen::<f32>();
            }
        }
    }
}

//////////////////////////////////
// Chromosome
//////////////////////////////////
#[derive(Clone, Debug)]
pub struct Chromosome {
    pub genes: Vec<f32>,
}

impl Chromosome {
    pub fn len(&self) -> usize {
        self.genes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.genes.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = &f32> {
        self.genes.iter()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut f32> {
        self.genes.iter_mut()
    }
}

impl Index<usize> for Chromosome {
    type Output = f32;

    fn index(&self, index: usize) -> &Self::Output {
        &self.genes[index]
    }
}

impl FromIterator<f32> for Chromosome {
    fn from_iter<T: IntoIterator<Item = f32>>(iter: T) -> Self {
        Self {
            genes: iter.into_iter().collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;
    use std::collections::BTreeMap;

    //////////////////////////////////
    // Test Individual
    //////////////////////////////////
    #[cfg(test)]
    #[derive(Clone, Debug, PartialEq)]
    pub enum TestIndividual {
        /// For tests that require access to chromosome
        WithChromosome { chromosome: Chromosome },

        /// For tests that don't require access to chromosome
        WithFitness { fitness: f32 },
    }

    #[cfg(test)]
    impl TestIndividual {
        pub fn new(fitness: f32) -> Self {
            Self::WithFitness { fitness }
        }
    }

    #[cfg(test)]
    impl Individual for TestIndividual {
        fn create(chromosome: Chromosome) -> Self {
            Self::WithChromosome { chromosome }
        }

        fn chromosome(&self) -> &Chromosome {
            match self {
                Self::WithChromosome { chromosome } => chromosome,
                Self::WithFitness { .. } => {
                    panic!("not supported for TestIndividual::WithFitness")
                }
            }
        }

        fn fitness(&self) -> f32 {
            match self {
                Self::WithChromosome { chromosome } => chromosome.iter().sum(),
                Self::WithFitness { fitness } => *fitness,
            }
        }
    }

    #[cfg(test)]
    impl PartialEq for Chromosome {
        fn eq(&self, other: &Self) -> bool {
            approx::relative_eq!(self.genes.as_slice(), other.genes.as_slice())
        }
    }

    //////////////////////////////////
    // Selection tests
    //////////////////////////////////
    #[test]
    fn roulette_selection() {
        let method = RouletteWheelSelection::new();
        let mut rng = ChaCha8Rng::from_seed(Default::default());

        let population = vec![
            TestIndividual::new(2.0),
            TestIndividual::new(1.0),
            TestIndividual::new(4.0),
            TestIndividual::new(3.0),
        ];

        let actual_histogram: BTreeMap<i32, _> = (0..1000)
            .map(|_| method.select(&mut rng, &population))
            .fold(Default::default(), |mut histogram, individual| {
                *histogram.entry(individual.fitness() as _).or_default() += 1;

                histogram
            });

        let expected_histogram = maplit::btreemap! {
            // fitness => how many times this fitness has been chosen
            1 => 98,
            2 => 202,
            3 => 278,
            4 => 422,
        };

        assert_eq!(actual_histogram, expected_histogram);
    }

    //////////////////////////////////
    // Chromosome tests
    //////////////////////////////////
    fn chromosome() -> Chromosome {
        Chromosome {
            genes: vec![3.0, 1.0, 2.0],
        }
    }

    #[test]
    fn chromosome_len() {
        assert_eq!(chromosome().len(), 3);
    }

    #[test]
    fn chromosome_iter() {
        let chromosome = chromosome();
        let genes: Vec<_> = chromosome.iter().collect();

        assert_eq!(genes.len(), 3);
        assert_eq!(genes[0], &3.0);
        assert_eq!(genes[1], &1.0);
        assert_eq!(genes[2], &2.0);
    }

    #[test]
    fn chromosome_iter_mut() {
        let mut chromosome = chromosome();

        chromosome.iter_mut().for_each(|gene| {
            *gene *= 10.0;
        });

        let genes: Vec<_> = chromosome.iter().collect();

        assert_eq!(genes.len(), 3);
        assert_eq!(genes[0], &30.0);
        assert_eq!(genes[1], &10.0);
        assert_eq!(genes[2], &20.0);
    }

    #[test]
    fn chromosome_index() {
        let chromosome = Chromosome {
            genes: vec![3.0, 1.0, 2.0],
        };

        assert_eq!(chromosome[0], 3.0);
        assert_eq!(chromosome[1], 1.0);
        assert_eq!(chromosome[2], 2.0);
    }

    #[test]
    fn chromosome_from_iter() {
        let chromosome: Chromosome = vec![3.0, 1.0, 2.0].into_iter().collect();

        assert_eq!(chromosome[0], 3.0);
        assert_eq!(chromosome[1], 1.0);
        assert_eq!(chromosome[2], 2.0);
    }

    //////////////////////////////////
    // Crossover tests
    //////////////////////////////////
    #[test]
    fn uniform_crossover() {
        let mut rng = ChaCha8Rng::from_seed(Default::default());
        let parent_a: Chromosome = (1..=100).map(|n| n as f32).collect();
        let parent_b: Chromosome = (1..=100).map(|n| -n as f32).collect();

        let child = UniformCrossover::new().crossover(&mut rng, &parent_a, &parent_b);

        // Number of genes different between `child` and `parent_a`
        let diff_a = child
            .iter()
            .zip(parent_a.iter())
            .filter(|(c, p)| *c != *p)
            .count();

        // Number of genes different between `child` and `parent_b`
        let diff_b = child
            .iter()
            .zip(parent_b.iter())
            .filter(|(c, p)| *c != *p)
            .count();

        assert_eq!(diff_a, 49);
        assert_eq!(diff_b, 51);
    }

    //////////////////////////////////
    // Mutation tests
    //////////////////////////////////
    fn actual(chance: f32, coeff: f32) -> Vec<f32> {
        let mut child = vec![1.0, 2.0, 3.0, 4.0, 5.0].into_iter().collect();

        let mut rng = ChaCha8Rng::from_seed(Default::default());

        GaussianMutation::new(chance, coeff).mutate(&mut rng, &mut child);

        child.iter().cloned().collect()
    }

    #[test]
    fn zero_chance_zero_coeff_no_change() {
        let actual = actual(0.0, 0.0);
        let expected = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        approx::assert_relative_eq!(actual.as_slice(), expected.as_slice(),);
    }

    #[test]
    fn zero_chance_non_zero_coeff_no_change() {
        let actual = actual(0.0, 0.5);
        let expected = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        approx::assert_relative_eq!(actual.as_slice(), expected.as_slice(),);
    }

    #[test]
    fn fifty_fifty_chance_zero_coeff_no_change() {
        let actual = actual(0.5, 0.0);
        let expected = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        approx::assert_relative_eq!(actual.as_slice(), expected.as_slice(),);
    }

    #[test]
    fn fifty_fifty_chance_non_zero_coeff_slight_change() {
        let actual = actual(0.5, 0.5);
        let expected = vec![1.0, 1.7756249, 3.0, 4.1596804, 5.0];

        approx::assert_relative_eq!(actual.as_slice(), expected.as_slice(),);
    }

    #[test]
    fn max_chance_zero_coeff_no_change() {
        let actual = actual(1.0, 0.0);
        let expected = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        approx::assert_relative_eq!(actual.as_slice(), expected.as_slice(),);
    }

    #[test]
    fn max_chance_non_zero_coeff_big_change() {
        let actual = actual(1.0, 0.5);

        let expected = vec![1.4545316, 2.1162078, 2.7756248, 3.9505124, 4.638691];

        approx::assert_relative_eq!(actual.as_slice(), expected.as_slice(),);
    }

    //////////////////////////////////
    // Evolve tests
    //////////////////////////////////
    fn individual(genes: &[f32]) -> TestIndividual {
        let chromosome = genes.iter().cloned().collect();

        TestIndividual::create(chromosome)
    }

    #[test]
    fn evolve() {
        let mut rng = ChaCha8Rng::from_seed(Default::default());

        let ga = GeneticAlgorithm::new(
            RouletteWheelSelection::new(),
            UniformCrossover::new(),
            GaussianMutation::new(0.5, 0.5),
        );

        let mut population = vec![
            individual(&[0.0, 0.0, 0.0]), // fitness = 0.0
            individual(&[1.0, 1.0, 1.0]), // fitness = 3.0
            individual(&[1.0, 2.0, 1.0]), // fitness = 4.0
            individual(&[1.0, 2.0, 4.0]), // fitness = 7.0
        ];

        for _ in 0..10 {
            (population, _) = ga.evolve(&mut rng, &population);
        }

        let expected_population = vec![
            individual(&[0.44769490, 2.0648358, 4.3058133]), // fitness ~= 6.8
            individual(&[1.21268670, 1.5538777, 2.8869110]), // fitness ~= 5.7
            individual(&[1.06176780, 2.2657390, 4.4287640]), // fitness ~= 7.8
            individual(&[0.95909685, 2.4618788, 4.0247330]), // fitness ~= 7.4
        ];

        assert_eq!(population, expected_population);
    }
}
