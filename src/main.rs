use std::fs::File;
use std::io::{BufReader, BufRead, Write};
use rand::Rng;
use rust_bert::pipelines::generation::{GPT2Generator, LanguageGenerator, GenerateConfig};
use std::collections::HashSet;
use tch::{Device, Cuda};
use rust_bert::gpt2::*;
use rust_bert::resources::{Resource, RemoteResource};
use std::path::Path;

fn main() {
    // use torch_sys::dummy_cuda_dependency;
    // unsafe { dummy_cuda_dependency(); }


    // this file contains one short and memorable phrase per line
    // see https://www.yorku.ca/mack/chi03b.html (accessed 1st May 2020)
    let lines = BufReader::new(File::open("phrases/mackenzie-soukoreff-phrases.txt").unwrap()).lines();

    // create a function that writes a line to the output file
    let mut output_line = {
        let output_file_name = format!("phrases/variation-triplets-xl-{}.txt", rand::thread_rng().gen::<u16>());
        assert!(!Path::new(&output_file_name).exists());

        let mut output_file = File::create(output_file_name).unwrap();

        move |line: &str| {
            output_file.write_all(line.as_bytes()).unwrap();
            output_file.write_all(b"\n").unwrap();
        }
    };

    // create the GPT-2 Model that generates our variations
    let mut model = GPT2Generator::new(GenerateConfig {

        // vary length from 2 to 10 to keep it short
        min_length: 3,
        max_length: 10,

        // enormous length penalty favors short sentences
        length_penalty: 1000.0,

        // always compute four variations at once
        num_return_sequences: 12,

        do_sample: true, // generates new random stuff each time
        temperature: 1.5,

        // top_k: 0,
        // top_p: 0.0,
        // early_stopping: false,
        // num_beams: 5,

        model_resource: Resource::Remote(RemoteResource::from_pretrained(Gpt2ModelResources::GPT2_XL)),
        merges_resource: Resource::Remote(RemoteResource::from_pretrained(Gpt2MergesResources::GPT2_XL)),
        vocab_resource: Resource::Remote(RemoteResource::from_pretrained(Gpt2VocabResources::GPT2_XL)),
        config_resource: Resource::Remote(RemoteResource::from_pretrained(Gpt2ConfigResources::GPT2_XL)),

        // device: Device::Cuda(0), // FIXME

        ..Default::default()

    }).unwrap();

    // generate 3 variations for each phrase in the file
    'line: for line in lines {
        let line = line.as_ref().unwrap().trim().to_string();

        // split the line into words to randomly remove the last few words
        let words = line.split_whitespace().collect::<Vec<&str>>();

        // skip phrases with less than five words
        if words.len() > 4 {
            println!("{}", line);
            let mut distinct_variations = HashSet::new();
            let mut iteration = 0;

            // randomly generate variations until we have three distinct ones
            while distinct_variations.len() < 3 {

                // cutoff at least one word from the end
                let phrase_base = words[ .. rand::thread_rng().gen_range(3, words.len() - 1) ].join(" ");

                // generate a few predictions at once, using the GTP-2 generator
                println!("generating variations for \"{}\"", phrase_base);
                let raw_variations = model.generate(Some(vec![&phrase_base]), None);

                // filter out incomplete sentences and cutoff everything after the first sentence
                let cleaned_variations = raw_variations.into_iter()
                    .filter_map(|sentence|{
                        if !sentence.chars().all(|c| c.is_alphanumeric() || " .!?,:-".contains(c)) {
                            return None;
                        }

                        let words = sentence.split_whitespace()
                            .map(String::from).collect::<Vec<_>>();

                        let sentence_end = words.iter().enumerate()
                            .find(|(_, word)| word.ends_with(|c| ".!?".contains(c)));

                        // if this variation contains a full sentence,
                        // cutoff the rest of the sentence (otherwise skip the whole phrase)
                        sentence_end.map(|(index, word)|{
                            let mut words = words[ .. index ].to_vec();
                            words.push(word.replace(".", ""));
                            words.join(" ")
                        })
                    })
                    .collect::<Vec<String>>();

                // collect the cleaned variations until we have 3
                for variation in cleaned_variations {
                    if distinct_variations.len() < 3 && variation != line {
                        if distinct_variations.insert(variation.clone()) {
                            println!("\t{}", variation);
                        }
                    }
                }

                // abort process if nothing can be found after 20 times
                iteration += 1;
                if iteration == 20 {
                    continue 'line;
                }
            }

            // save variations to file if we have 3
            output_line(&line);
            for variation in distinct_variations {
                output_line(&variation);
            }

            println!()
        }

    }
}
