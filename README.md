# EKS Deployment

This project outlines the infrastructure setup for a web application with a frontend and a backend component, using Kubernetes for orchestration and Docker for containerization. 

## Overview

The project is divided into two main parts:

1. **Backend Component**: A Flask application serving as the backend.
2. **Frontend Component**: A React application serving as the frontend.

Both components are containerized using Docker and are orchestrated to run on Kubernetes.

## Prerequisites

- Docker
- Kubernetes cluster (Minikube, EKS, GKE, etc.)
- kubectl configured to communicate with your cluster
- An AWS ECR (Elastic Container Registry) account or another container registry account

## Deployment Instructions

### Docker Images

The Dockerfiles included in the project setup the environments for both the frontend and backend components.

#### Backend Dockerfile

- Based on `python:3.8-slim`
- Installs necessary dependencies and copies the application code to the container
- Exposes port `5000`
- Runs `app.py` to start the Flask application

#### Frontend Dockerfile

- Based on `node:14-alpine3.14`
- Sets up the working directory and copies the application code
- Installs dependencies using `npm install`
- Exposes port `3000`
- Starts the application using `npm start`

### Kubernetes Objects

#### Backend Deployment and Service

- Deploys 2 replicas of the backend Flask application
- Uses the Docker image `public.ecr.aws/f5k2n2p7/my-flask-app:latest-amd64`
- Exposes the application internally through a Kubernetes service on TCP port 80, targeting container port `5000`

#### Frontend Deployment and Service

- Deploys 2 replicas of the frontend React application
- Uses the Docker image `public.ecr.aws/f5k2n2p7/segmentor-app:latest-amd64-v2`
- Exposes the application externally through a Kubernetes service on TCP port 80, targeting container port `3000`

### Deploying to Kubernetes

1. **Create the Docker Images**: Build the Docker images for both frontend and backend, and push them to your container registry.

2. **Apply Kubernetes Manifests**: Use `kubectl apply -f <filename>` to deploy the objects defined in the YAML files to your Kubernetes cluster.

   Example:
   ```
    kubectl apply -f backend-deployment.yaml
    kubectl apply -f backend-service.yaml
    kubectl apply -f frontend-deployment.yaml
    kubectl apply -f frontend-service.yaml
   ```

3. **Access the Services**: If you're using Minikube, you can access the services via `minikube service <service-name>`. For cloud-based clusters, the `LoadBalancer` service type will provision an external IP or hostname to access the service.

## Notes

- Ensure your Kubernetes cluster has sufficient resources to accommodate the resource limits defined in the deployments.
- Modify the image URLs in the deployment manifests to point to your container registry where the Docker images are hosted.
- The application ports are hardcoded (5000 for backend, 3000 for frontend). If you change these ports in your Dockerfiles, ensure you update the Kubernetes service definitions accordingly.

This README provides a basic outline for deploying a simple web application using Docker and Kubernetes. Adjust the configurations as necessary to fit your project's needs.
