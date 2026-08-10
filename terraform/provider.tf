terraform {
  backend "gcs" {
    bucket = "agent-force-tfstate"
    prefix = "terraform/state"
  }

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 6.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = "asia-southeast1"
}

variable "project_id" {
  description = "GCP project ID"
  type        = string
  default     = "project-440503b6-79af-45e1-ab2"
}
