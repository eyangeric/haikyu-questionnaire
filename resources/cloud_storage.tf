resource "google_storage_bucket" "ml_bucket" {
  name          = "character-ml-model"
  location      = "us-central1"
  force_destroy = true
}