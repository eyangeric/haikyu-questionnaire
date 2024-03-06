resource "google_storage_bucket" "ml_bucket" {
  name          = "character-ml-model"
  location      = "US"
  force_destroy = true
}