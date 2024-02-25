# Differential pseudo-gene activity analysis:
# ############################

# Load required libraries
library(ArchR)
library(BiocGenerics)
library(ComplexHeatmap)
library(circlize)
library(viridis)

######################################################################################
# ArchR
########################################################################################

# DEGs using ATAC labels
archr_atac_markers <- getMarkerFeatures(
  ArchRProj = proj,
  useMatrix = "ArchRGeneScore",
  groupBy = "ATAC_clusters",
  bias = c("TSSEnrichment", "log10(nFrags)"),
  testMethod = "wilcoxon"
)

# Create and draw heatmap for ATAC clusters
archr_atac_heatmap <- markerHeatmap(
  seMarker = archr_atac_markers,
  cutOff = "FDR <= 0.01 & Log2FC >= 1.25",
  labelMarkers = NULL,
  transpose = FALSE,
  pal = viridis(n = 256),
  limits = c(-2, 2)
)
ComplexHeatmap::draw(archr_atac_heatmap, heatmap_legend_side = "bot", annotation_legend_side = "bot")
plotPDF(archr_atac_heatmap, name = "GeneScores-Marker-Heatmap_ArchR", width = 8, height = 6, ArchRProj = proj, addDOC = FALSE)

# DEGs using predicted labels (removing small groups)
idx_sample <- BiocGenerics::which(proj$predictedScore_ArchR > 0.5)
cells_sample <- proj$cellNames[idx_sample]
proj_filter <- proj[cells_sample, ]

popular_groups <- summary(factor(proj_filter$predictedGroup_ArchR))
popular_groups <- popular_groups[popular_groups > 10]
proj_filter$Mode_Label <- ifelse(proj_filter$predictedGroup_ArchR %in% names(popular_groups), TRUE, FALSE)

idx_sample <- BiocGenerics::which(proj_filter$Mode_Label == TRUE)
cells_sample <- proj_filter$cellNames[idx_sample]
proj_filter <- proj_filter[cells_sample, ]

# DEGs using predicted labels
archr_pred_labels_markers <- getMarkerFeatures(
  ArchRProj = proj_filter,
  useMatrix = "ArchRGeneScore",
  groupBy = "predictedGroup_ArchR",
  bias = c("TSSEnrichment", "log10(nFrags)"),
  testMethod = "wilcoxon"
)
archr_pred_labels_heatmap <- markerHeatmap(
  seMarker = archr_pred_labels_markers,
  cutOff = "FDR <= 0.01 & Log2FC >= 1.25",
  labelMarkers = NULL,
  transpose = FALSE,
  pal = viridis(n = 256),
  limits = c(-2, 2)
)
ComplexHeatmap::draw(archr_pred_labels_heatmap, heatmap_legend_side = "bot", annotation_legend_side = "bot")
plotPDF(archr_pred_labels_heatmap, name = "GeneScores-Marker-Heatmap_ArchR_pred", width = 8, height = 6, ArchRProj = proj_filter, addDOC = FALSE)

# Differential peak analysis:
############################

# ATAC clusters
proj <- addGroupCoverages(ArchRProj = proj, groupBy = "ATAC_clusters", force = TRUE)

path_to_macs2 <- findMacs2()

proj <- addReproduciblePeakSet(
  ArchRProj = proj,
  groupBy = "ATAC_clusters",
  pathToMacs2 = path_to_macs2,
  force = TRUE
)
proj <- addPeakMatrix(proj, force = TRUE)
proj <- addBgdPeaks(proj)

# DEGs using peaks for ATAC clusters
peaks_markers <- getMarkerFeatures(
  ArchRProj = proj,
  useMatrix = "PeakMatrix",
  groupBy = "ATAC_clusters",
  bias = c("TSSEnrichment", "log10(nFrags)"),
  testMethod = "wilcoxon"
)

# Create and draw heatmap for peaks in ATAC clusters
peaks_heatmap <- markerHeatmap(
  seMarker = peaks_markers,
  cutOff = "FDR <= 0.01 & Log2FC >= 1.25",
  labelMarkers = NULL,
  transpose = FALSE,
  pal = viridis(n = 256),
  limits = c(-2, 2)
)
ComplexHeatmap::draw(peaks_heatmap, heatmap_legend_side = "bot", annotation_legend_side = "bot")
plotPDF(peaks_heatmap, name = "Markers_peaks_ATAC_clusters", width = 8, height = 6, ArchRProj = proj, addDOC = FALSE)

# ArchR predicted labels

# DEGs using predicted labels (removing small groups)
idx_sample <- BiocGenerics::which(proj$predictedScore_ArchR >= 0.5)
cells_sample <- proj$cellNames[idx_sample]
proj_filter <- proj[cells_sample, ]

popular_groups <- summary(factor(proj_filter$predictedGroup_ArchR))
popular_groups <- popular_groups[popular_groups > 10]
proj_filter$Mode_Label <- ifelse(proj_filter$predictedGroup_ArchR %in% names(popular_groups), TRUE, FALSE)

idx_sample <- BiocGenerics::which(proj_filter$Mode_Label == TRUE)
cells_sample <- proj_filter$cellNames[idx_sample]
proj_filter <- proj_filter[cells_sample, ]

proj_archr <- addGroupCoverages(ArchRProj = proj_filter, groupBy = "predictedGroup_ArchR", force = TRUE)

path_to_macs2 <- findMacs2()

proj_archr <- addReproduciblePeakSet(
  ArchRProj = proj_archr,
  groupBy = "predictedGroup_ArchR",
  pathToMacs2 = path_to_macs2,
  force = TRUE
)
proj_archr <- addPeakMatrix(proj_archr, force = TRUE)
proj_archr <- addBgdPeaks(proj_archr, force = TRUE)

# DEGs using peaks for predicted labels
archr_pred_labels_peaks_markers <- getMarkerFeatures(
  ArchRProj = proj_archr,
  useMatrix = "PeakMatrix",
  groupBy = "predictedGroup_ArchR",
  bias = c("TSSEnrichment", "log10(nFrags)"),
  testMethod = "wilcoxon"
)

# Create and draw heatmap for peaks in predicted labels
archr_pred_labels_peaks_heatmap <- markerHeatmap(
  seMarker = archr_pred_labels_peaks_markers,
  cutOff = "FDR <= 0.01 & Log2FC >= 1.25",
  labelMarkers = NULL,
  transpose = FALSE,
  pal = viridis(n = 256),
  limits = c(-2, 2)
)
ComplexHeatmap::draw(archr_pred_labels_peaks_heatmap, heatmap_legend_side = "bot", annotation_legend_side = "bot")
plotPDF(archr_pred_labels_peaks_heatmap, name = "Markers_peaks_Archr_Predicted_labels", width = 8, height = 6, ArchRProj = proj_archr, addDOC = FALSE)

# Save the ArchR project with predicted labels
saveRDS(proj_archr, "./final_archr_proj_archrGS.rds
