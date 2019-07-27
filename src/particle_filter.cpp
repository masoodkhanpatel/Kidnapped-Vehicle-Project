#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;
using std::normal_distribution;
using std::uniform_int_distribution;
using std::uniform_real_distribution;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
 
  num_particles = 100;  // the number of particles

  particles = vector<Particle>(num_particles);

  std::default_random_engine gen;
  double std_x, std_y, std_theta;  // Standard deviations

  std_x = std[0];
  std_y = std[1];
  std_theta = std[2];

  // Create Gaussian distributions
  normal_distribution<double> dist_x(x, std_x);
  normal_distribution<double> dist_y(y, std_y);
  normal_distribution<double> dist_theta(theta, std_theta);

  for (int i = 0; i < num_particles; ++i) {
    Particle particle;

    particle.x = dist_x(gen);
    particle.y = dist_y(gen);
    particle.theta = dist_theta(gen);
    particle.weight = 1;

    particles[i] = particle;
  }

  weights = vector<double >(num_particles, 1);
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
 
  std::default_random_engine gen;
  double std_x, std_y, std_theta;

  for (Particle p : particles) {
    std_x = std_pos[0];
    std_y = std_pos[1];
    std_theta = std_pos[2];

    double delta_theta = yaw_rate*delta_t;

    double predicted_x = p.x + (velocity/yaw_rate)*(sin(p.theta+delta_theta) - sin(p.theta));
    double predicted_y = p.y + (velocity/yaw_rate)*(cos(p.theta) - cos(p.theta+delta_theta));
    double predicted_theta = p.theta + delta_theta;

    // Add noise
    normal_distribution<double> dist_x(predicted_x, std_x);
    normal_distribution<double> dist_y(predicted_y, std_y);
    normal_distribution<double> dist_theta(predicted_theta, std_theta);

    p.x = dist_x(gen);
    p.y = dist_y(gen);
    p.theta = dist_theta(gen);
  }
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  
  for(LandmarkObs p: predicted) {
    double minDistance = -1;
    double currentDistance = -1;

    for (LandmarkObs o: observations) {
      currentDistance = dist(p.x, p.y, o.x, o.y);

      if (minDistance == -1 || currentDistance < minDistance) {
        minDistance = currentDistance;
        o.id = p.id;
      }
    }
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  
  double weight_normalizer = 0.0;

  for (int i = 0; i < num_particles; ++i) {
    double particle_x = particles[i].x;
    double particle_y = particles[i].y;
    double particle_theta = particles[i].theta;

    // Transformed observations from local reference map to global reference map
    vector<LandmarkObs> transformed_observations;

    for (const auto & observation : observations) {
      LandmarkObs transformed_obs;
      transformed_obs.id = observation.id;
      transformed_obs.x = particle_x + (cos(particle_theta) * observation.x) - (sin(particle_theta) * observation.y);
      transformed_obs.y = particle_y + (sin(particle_theta) * observation.x) + (cos(particle_theta) * observation.y);
      transformed_observations.push_back(transformed_obs);
    }

    // Considering observations within range
    vector<LandmarkObs> predicted_landmarks;
    for (auto current_landmark : map_landmarks.landmark_list) {
      if (dist(particle_x, particle_y, current_landmark.x_f, current_landmark.y_f) <= sensor_range) {
        predicted_landmarks.push_back(LandmarkObs {current_landmark.id_i, current_landmark.x_f, current_landmark.y_f});
      }
    }

    dataAssociation(predicted_landmarks, transformed_observations);

    for (int k = 0; k < transformed_observations.size(); ++k) {
      double sig_x, sig_y;
      sig_x = std_landmark[0];
      sig_y = std_landmark[1];
      double gauss_norm = 1 / (2 * M_PI * sig_x * sig_y);
      double exponent;
      double x_tr_obs, y_tr_obs, mu_x, mu_y;
      x_tr_obs = transformed_observations[k].x;
      y_tr_obs = transformed_observations[k].y;

      for (int l = 0; l < map_landmarks.landmark_list.size(); l++) {
        if(map_landmarks.landmark_list[l].id_i == transformed_observations[k].id) {
          mu_x = predicted_landmarks[l].x;
          mu_y = predicted_landmarks[l].y;

          exponent = (pow(x_tr_obs - mu_x, 2) / (2 * pow(sig_x, 2)))
                     + (pow(y_tr_obs - mu_y, 2) / (2 * pow(sig_y, 2)));

          particles[i].weight *= gauss_norm * exp(-exponent);
        }
      }
      weights[k] = gauss_norm * exp(-exponent);
    }
    weight_normalizer += particles[i].weight;
  }

  for (int i = 0; i < particles.size(); i++) {
    particles[i].weight /= weight_normalizer;
    weights[i] = particles[i].weight;
  }
}

void ParticleFilter::resample() {
  
  vector<Particle> resampled_particles;
  std::default_random_engine gen;
  uniform_int_distribution<int> particle_index(0, num_particles - 1);

  int current_index = particle_index(gen);
  double beta = 0.0;
  double max_weight = *max_element(std::begin(weights), std::end(weights));

  for (int i = 0; i < particles.size(); i++) {
    uniform_real_distribution<double> random_weight(0.0, max_weight * 2);
    beta += random_weight(gen);

    while (beta > weights[current_index]) {
      beta -= weights[current_index];
      current_index = (current_index + 1) % num_particles;
    }
    resampled_particles.push_back(particles[current_index]);
  }
  particles = resampled_particles;
}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}