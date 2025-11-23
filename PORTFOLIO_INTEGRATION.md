# SkinCBM Portfolio Integration Guide

This document describes how SkinCBM has been integrated into the portfolio website at matt-cockayne.github.io.

## Files Added to Portfolio

### 1. Project Page
**Location**: `_projects/skincbm.md`
- Complete project description
- Technical architecture details
- Visual examples with 8 visualization images
- 3 interactive notebook tutorials with collapsible iframe previews
- Clinical significance and 7-point checklist explanation
- Quick start guide and code examples

### 2. Visualization Images
**Location**: `assets/projects/skincbm/`

**Individual Case Demos:**
- `demo_case_7.png` (519KB) - Basal Cell Carcinoma demo
- `demo_case_578.png` (520KB) - Melanoma with high 7-point score
- `demo_case_596.png` (533KB) - Melanoma with moderate features
- `demo_case_657.png` (535KB) - Additional melanoma case

**Individual Case Interventions:**
- `intervention_case_7.png` (518KB) - Intervention analysis for case 7
- `intervention_case_578.png` (523KB) - Intervention analysis for case 578
- `intervention_case_596.png` (532KB) - Intervention analysis for case 596
- `intervention_case_657.png` (545KB) - Intervention analysis for case 657

**Systematic Intervention Analysis (across full dataset):**
- `1_performance_comparison.png` (223KB) - Original vs corrected model performance
- `2_concept_impact_analysis.png` (252KB) - Relative importance of each concept
- `3_intervention_direction_analysis.png` (122KB) - Asymmetric intervention effects (0→1 vs 1→0)
- `4_confusion_matrices.png` (51KB) - Before/after confusion matrix comparison

**Total**: 12 images, ~4.8MB

### 3. HTML Notebooks
**Location**: `assets/notebooks/skincbm/`
- `01_cbm_training_walkthrough.html` (2.1MB) - Complete training pipeline
- `02_demo_with_sample_data.html` (943KB) - Quick demo with 4 sample cases
- `03_demo_intervention.html` (1.1MB) - Concept intervention tutorial

**Total**: 3 notebooks, ~4.1MB

### 4. Projects Page Update
**Location**: `projects.md`
- Added SkinCBM card in PhD Research Projects section (now 4 projects)
- Added SkinCBM Tutorials card in Interactive Demonstrations section
- Updated Research Impact statistics:
  - Changed "1 Open-Source Toolkit" to "2 Open-Source Toolkits"
  - Updated tutorial count from "6+" to "9+"

## Project Page Structure

The SkinCBM project page (`_projects/skincbm.md`) includes:

### Sections
1. **Overview** - Introduction to Concept Bottleneck Models
2. **Key Features** - Interpretable architecture, intervention, information theory, educational focus
3. **Visual Examples** - 2x2 grid with demo and intervention examples
4. **Interactive Tutorials** - 3 collapsible sections with embedded notebooks
5. **Technical Details** - Architecture, training strategies, performance metrics
6. **Quick Start** - Installation and usage examples
7. **Repository Structure** - Complete directory tree
8. **Clinical Significance** - Why interpretability matters, 7-point checklist
9. **Documentation** - Links to detailed guides
10. **Research Context** - Related publications and novel contributions
11. **Future Directions** - Planned enhancements and research questions
12. **Citation** - BibTeX entry
13. **Resources** - Links to GitHub, tutorials, sample data

### Embedded Notebooks
Each tutorial has:
- Collapsible `<details>` section with summary
- Embedded iframe (600px height) displaying full HTML notebook
- Brief description of tutorial content
- Link to GitHub for original .ipynb file

### Visual Grid
Images displayed in 2-column responsive grid with:
- Rounded corners and subtle shadows
- Descriptive captions
- Auto-fit layout (min 300px per column)

## Deployment Instructions

### Current Status
All files are staged in the local repository. To deploy:

```bash
cd /home/matthewcockayne/Documents/PhD/portfolio/matt-cockayne.github.io

# Check current branch and status
git status

# Add all new files
git add _projects/skincbm.md
git add assets/projects/skincbm/
git add assets/notebooks/skincbm/
git add projects.md

# Commit changes
git commit -m "Add SkinCBM project: Concept Bottleneck Models for interpretable diagnosis"

# Push to GitHub
git push origin test
```

### Verify Deployment
After GitHub Pages rebuilds (~2-3 minutes):
1. Visit https://matt-cockayne.github.io/projects
2. Check SkinCBM card appears in PhD Research Projects
3. Click "View Project →" button
4. Verify all 8 images load correctly
5. Test collapsible notebook sections
6. Confirm iframe notebooks display properly
7. Check all GitHub links work

## Key Differences from MedXAI Integration

### Similarities
- Same overall structure (overview, features, visuals, tutorials, technical details)
- Collapsible notebook sections with iframes
- 2-column image grid
- Links to GitHub repository

### Differences
1. **Focus**: SkinCBM emphasizes interpretability through concepts vs MedXAI's post-hoc explainability
2. **Clinical Content**: Includes 7-point checklist explanation and clinical significance section
3. **Notebooks**: 3 tutorials (vs 6 for MedXAI) but includes training walkthrough
4. **Sample Data**: Highlights that sample data is included (no dataset required for demos)
5. **Information Theory**: Emphasizes novel completeness and synergy metrics
6. **Research Context**: Includes "Novel Contributions" and "Future Directions" sections

## Repository Statistics

### Total Portfolio Assets
After SkinCBM integration:
- **Projects**: 4 (DermFormer, SkinCBM, MedXAI, Classification-to-Segmentation)
- **Visualizations**: 18 images (~8.0MB total)
- **Notebook Tutorials**: 9 notebooks (~10MB total)
- **Interactive Demos**: 4 demonstration interfaces

### SkinCBM Specific
- **Code Repository**: https://github.com/Matt-Cockayne/SynergyCBM/tree/main/SkinCBM
- **Documentation Pages**: 4 (INSTALLATION, QUICKSTART, ARCHITECTURE, DATASETS)
- **Example Scripts**: 4 (train, demo, intervention, analysis)
- **Sample Cases**: 4 dermoscopy cases with metadata
- **Model Checkpoints**: Training outputs stored locally

## Maintenance Notes

### Updating Visualizations
If new visualization outputs are generated:
```bash
cd /home/matthewcockayne/Documents/PhD/portfolio/SkinCBM/notebooks/outputs
# Generate new visualizations
# Then copy to portfolio:
cp *.png /home/matthewcockayne/Documents/PhD/portfolio/matt-cockayne.github.io/assets/projects/skincbm/
```

### Updating Notebooks
If notebooks are modified:
```bash
cd /home/matthewcockayne/Documents/PhD/portfolio/SkinCBM/notebooks
# Convert to HTML
jupyter nbconvert --to html *.ipynb
# Copy to portfolio
cp *.html /home/matthewcockayne/Documents/PhD/portfolio/matt-cockayne.github.io/assets/notebooks/skincbm/
```

### Adding New Tutorials
1. Create notebook in `SkinCBM/notebooks/`
2. Convert to HTML: `jupyter nbconvert --to html new_tutorial.ipynb`
3. Copy HTML to portfolio: `cp new_tutorial.html matt-cockayne.github.io/assets/notebooks/skincbm/`
4. Add collapsible section to `_projects/skincbm.md`:
```markdown
### N. Tutorial Title
Description of tutorial

<details>
<summary><strong>View Tutorial</strong> (Click to expand)</summary>
<iframe src="/assets/notebooks/skincbm/new_tutorial.html" width="100%" height="600px"></iframe>
<p><em>Tutorial covers: topics here</em></p>
<p><a href="GITHUB_LINK" target="_blank">View on GitHub →</a></p>
</details>
```

## SEO and Discoverability

### Meta Tags
The project page includes:
- **Title**: "SkinCBM: Concept Bottleneck Models for Interpretable Medical Diagnosis"
- **Description**: "Educational implementation of Concept Bottleneck Models..."
- **Tags**: Explainable AI, Medical Imaging, Interpretability, Deep Learning, Dermatology
- **Technologies**: PyTorch, Python, Jupyter, ResNet

### Search Keywords
Optimized for:
- Concept Bottleneck Models
- Interpretable AI
- Medical image analysis
- Dermatology AI
- Explainable medical diagnosis
- 7-point checklist
- Concept intervention

## Analytics Tracking

### Key Metrics to Monitor
- Page views on `/projects/skincbm`
- Clicks on "View Project" button from projects.md
- Notebook iframe engagement
- GitHub repository traffic from portfolio
- Time spent on page
- Scroll depth (tutorial engagement)

### Conversion Goals
- Clicks to GitHub repository
- Notebook collapsible expansions
- Downloads/clones of repository
- Tutorial completions

## Accessibility

### Implemented Features
- Semantic HTML structure
- Alt text for images (descriptive captions)
- Keyboard navigation for collapsible sections
- Sufficient color contrast
- Responsive design for mobile devices
- Clear heading hierarchy

### Future Improvements
- Add ARIA labels for interactive elements
- Transcript alternatives for visual content
- Screen reader testing
- Mobile-specific optimization

---

**Integration Date**: November 23, 2025  
**Portfolio Version**: v2.0 (4 projects integrated)  
**Status**: Ready for deployment  
