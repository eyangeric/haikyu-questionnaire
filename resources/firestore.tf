resource "google_firestore_database" "haikyu_database" {
  name = "haikyu"
  project = "haikyu-questionnaire"
  location_id = "us-central1"
  type = "FIRESTORE_NATIVE"
}