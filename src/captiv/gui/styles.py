css = """
.main {
  flex-shrink: 0;
  flex-grow: 1;
}

.main .body .gallery .thumbnails {
  padding-left: 100px;
  padding-right: 10px;
}

.main .body .gallery .gallery-container {
  height: 100%;
  overflow-y: auto !important;
  scrollbar-color:
  scrollbar-width: thin;
}

.main .body .gallery .gallery-container::-webkit-scrollbar {
  width: 8px;
}

.main .body .gallery .gallery-container::-webkit-scrollbar-track {
  background: transparent; /* Chrome/Safari/Edge */
}

.main .body .gallery .gallery-container::-webkit-scrollbar-thumb {
  background-color:
  border-radius: 4px;
}

.main .body .gallery .gallery-container .fixed-height {
  max-height: 0;
}

.main .body .gallery .gallery-container .grid-wrap {
  overflow: visible;
}

.accordion {
  padding: 0;
}

.accordion > button {
  padding: var(--block-padding);
}

.accordion .form {
  border: none;
}

.accordion .form .field {
  border: none;
}

.accordion fieldset {
  padding: 0 12px;
}
"""
