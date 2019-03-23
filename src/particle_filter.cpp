/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

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

#define EPS 0.000001

using std::string;
using std::vector;
using std::normal_distribution;

void ParticleFilter::init(double x, double y, double theta, double std[]) {

  // Initializing the particle number:
  num_particles = 100;

  // Adjusting the weight vector and particle vector sizes:
  particles.resize(num_particles);
  weights.resize(num_particles);

  // Extracting the standard deviations:
  double std_x = std[0];
  double std_y = std[1];
  double std_theta = std[2];

  // Creating a normal distribution for each parameter with its mean (GPS estimate) and std (given uncertainty):
  normal_distribution<double> dist_x(x, std_x);
  normal_distribution<double> dist_y(y, std_y);
  normal_distribution<double> dist_theta(theta, std_theta);

  // Initializing each of the particles via sampling the normal distributions:
  for (int i = 0; i < num_particles; ++i) {
    particles[i].id = i;
    particles[i].x = dist_x(gen);
    particles[i].y = dist_y(gen);
    particles[i].theta = dist_theta(gen);
    particles[i].weight = 1.0;
  }

  // Done initializing:
  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[],
                                double velocity, double yaw_rate) {

  // Extracting the standard deviations:
  double std_x = std_pos[0];
  double std_y = std_pos[1];
  double std_theta = std_pos[2];

  // Creating a normal distribution for each parameter with its mean (GPS estimate) and std (given uncertainty):
  normal_distribution<double> dist_x(0.0, std_x);
  normal_distribution<double> dist_y(0.0, std_y);
  normal_distribution<double> dist_theta(0.0, std_theta);

  // Updating the new position and heading of each particle:
  for (int i = 0; i < num_particles; ++i) {
    double theta_0 = particles[i].theta;

    // Yaw rate is roughly zero:
    if (fabs(yaw_rate) < EPS) {
      particles[i].x += velocity * cos(theta_0) * delta_t;
      particles[i].y += velocity * sin(theta_0) * delta_t;
    }
    else {
      particles[i].x += (velocity/yaw_rate) * (sin(theta_0+yaw_rate*delta_t) - sin(theta_0));
      particles[i].y += (velocity/yaw_rate) * (cos(theta_0) - cos(theta_0 + yaw_rate*delta_t));
      particles[i].theta += yaw_rate * delta_t;
    }

    // Adding random Gaussian noise to the particle's new position and heading:
    particles[i].x += dist_x(gen);
    particles[i].y += dist_y(gen);
    particles[i].theta += dist_theta(gen);
  }
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted,
                                     vector<LandmarkObs>& observations) {

  // Iterating over all of the observations:
  for (unsigned int i = 0; i < observations.size(); ++i) {

    // Initializing minimum distance to a very large number:
    double min_dist = std::numeric_limits<double>::infinity();

    // Initializing the nearest landmark id to something that doesn't exist on the map:
    int nearest_landmark_id = -1;

    // Iterating over each predicted landmark to find the nearest:
    for (unsigned int j = 0; j < predicted.size(); ++j) {
      // Calculating euclidean distance to the landmark:
      double euc_distance = dist(observations[i].x, observations[i].y, predicted[j].x, predicted[j].y);

      // Updating minimum distance & landmark id on finding closer landmark:
      if (euc_distance < min_dist) {
        min_dist = euc_distance;
        nearest_landmark_id = j;
      }
    }

    // Associate the observation to the nearest landmark:
    observations[i].id = nearest_landmark_id;
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   const vector<LandmarkObs> &observations,
                                   const Map &map_landmarks) {
  // Extracting the standard deviations:
  double std_x = std_landmark[0];
  double std_y = std_landmark[1];

  // Creating a covariance variable for each coordinate:
  double cov_x = std_x * std_x;
  double cov_y = std_y * std_y;

  // Calculating the (static) gaussian normalizer:
  double gauss_norm = 1 / (2.0 * M_PI * std_x * std_y);

  // Iterate over each particle for weight update:
  for (int i = 0; i < num_particles; ++i) {

    // Extracting a particle's coordinates and heading:
    double p_x = particles[i].x;
    double p_y = particles[i].y;
    double p_theta = particles[i].theta;

    // Locating landmarks limited by sensor range:
    vector<LandmarkObs> predicted_landmarks;
    for (unsigned int j = 0; j < map_landmarks.landmark_list.size(); ++j) {
      LandmarkObs measurement;

      // Extract each map landmark's id & (x,y) coordinates:
      int landmark_id = map_landmarks.landmark_list[j].id_i;
      float landmark_x = map_landmarks.landmark_list[j].x_f;
      float landmark_y = map_landmarks.landmark_list[j].y_f;

      // Finding the euclidean distance between the particle's location and the landmarks:
      double euc_dist = dist(p_x, p_y, landmark_x, landmark_y);

      // Accept landmarks that are within sensor range:
      if (euc_dist <= sensor_range) {
        measurement.id = landmark_id;
        measurement.x = landmark_x;
        measurement.y = landmark_y;
        predicted_landmarks.push_back(measurement);
      }
    }

    // Observation transformation to map coordinate system:
    vector<LandmarkObs> map_observations;
    for (unsigned int j = 0; j < observations.size(); ++j) {
      LandmarkObs map_obs;

      LandmarkObs obs = observations[j];
      double map_x = p_x + (cos(p_theta) * obs.x) - (sin(p_theta) * obs.y);
      double map_y = p_y + (sin(p_theta) * obs.x) + (cos(p_theta) * obs.y);

      // Assigning the transformed coordinates and id:
      map_obs.id = obs.id;
      map_obs.x = map_x;
      map_obs.y = map_y;

      map_observations.push_back(map_obs);
    }

    // Associate transformed observations with a predicted landmark:
    dataAssociation(predicted_landmarks, map_observations);

    // Weight calculation using the Multivariate-Gaussian distribution:
    double particle_weight = 1.0;  // Total weight of a particle

    for (unsigned int j = 0; j < map_observations.size(); ++j) {
      LandmarkObs map_obs = map_observations[j];
      LandmarkObs pred_landmark = predicted_landmarks[map_obs.id];

      double dx = map_obs.x - pred_landmark.x;
      double dy = map_obs.y - pred_landmark.y;
      double dx_2 = dx * dx;
      double dy_2 = dy * dy;
      double gauss_exp = exp(-( (dx_2 / (2*cov_x)) + (dy_2 / (2*cov_y))));

      particle_weight *= gauss_norm * gauss_exp;
    }

    // Assigning each particle to its new weight:
    particles[i].weight = particle_weight;
    weights[i] = particle_weight;
  }
}

void ParticleFilter::resample() {

  // Initializing a new particles vector;
  vector<Particle> resampled_particles(num_particles);

  // Create a discrete distributions - weight dependant:
  std::discrete_distribution<int> w_index(weights.begin(), weights.end());

  // Resampling particles with regards to their weights by sampling the discrete distribution:
  for (int i = 0; i < num_particles; ++i) {
    resampled_particles[i] = particles[w_index(gen)];
  }

  // Old becomes new - changing our particles vector to the re-sampled one:
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