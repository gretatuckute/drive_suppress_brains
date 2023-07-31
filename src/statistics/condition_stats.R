library(tidyverse)
library(lme4)
library(lmerTest)
library(MuMIn)
library(emmeans)
library(modelsummary)

##### SETTINGS #####
DATADIR = '/Users/gt/Documents/GitHub/drive_suppress_brains/data/'
DATAFNAME = 'brain-lang-data_participant_20230728.csv'
SAVEDIR = '/Users/gt/Documents/GitHub/drive_suppress_brains/src/statistics/condition_stats/' # Define the directory where stats will be saved
SAVESUBDIR = 'raw_R_output/'

# Check if the main SAVEDIR directory exists
if (!dir.exists(SAVEDIR)) {
  dir.create(SAVEDIR)
}

sub_dir_path <- file.path(SAVEDIR, SAVESUBDIR)

# Check if the subdirectory exists
if (!dir.exists(sub_dir_path)) {
  dir.create(sub_dir_path)
}

search_UIDs <- c('797', '841', '880')
modify_UIDs <- c('837', '856')

##### LME FORMULA #####
# Possible random intercepts are: target_UID, sessionindicator, run_within_session, item_id, trial_num
formula_full <- response_target ~ cond_factor + (1 | item_id_factor) + run_within_session + trial_num  

##### LOAD DATA #####
data_csv = read.csv(paste(DATADIR, DATAFNAME, sep=''), header=TRUE)

length(unique(data_csv$target_UID))
length(unique(data_csv$roi))

# Factorize variables
data_csv$cond_factor <- data_csv$cond
data_csv$cond_factor <- factor(data_csv$cond_factor, levels=c('B', 'D', 'S'))

# Convert item_id into a factor because we do not care which item_id is which 
data_csv$item_id_factor <- factor(data_csv$item_id)

########## CONDITION STATS FOR SEARCH: 797, 841, 880 ##########
data_search <- subset(data_csv, (target_UID %in% search_UIDs))

# Subset data
roi_of_interest = 'lang_LH_netw'
data_search_roi <- subset(data_search, (roi==roi_of_interest))

fname_save_search = paste(DATAFNAME,'_',roi_of_interest,'_','search.docx',sep="")

# Run LME
l.search <- lmer(formula_full, data=data_search_roi, REML=FALSE)
print(summary(l.search))
r.squaredGLMM(l.search)

modelsummary::modelsummary(l.search, fmt=3, stars=TRUE, 
                           coef_rename = c("cond_factorD" = "condition_D", "cond_factorB" = "condition_S", "item_id_factor" = "sentence", "trial_num" = "trial_within_run"),
                           statistic = c("SE = {std.error}", "t = {statistic}","p = {p.value}", "df = {df.error}"), 
                           output = paste(SAVEDIR, SAVESUBDIR, fname_save_search, sep=''))

# Run pairwise comparisons. PASTE THESE TO A COPY OF THE WORD DOC THAT WAS JUST GENERATED.
l.search.res <- summary(pairs(emmeans(l.search, "cond_factor", lmerTest.limit = 125000, pbkrtest.limit = 125000), reverse = FALSE)) # Get drive - suppress
l.search.res.rev <- summary(pairs(emmeans(l.search, "cond_factor", lmerTest.limit = 125000, pbkrtest.limit = 125000), reverse = TRUE)) # Get baseline

# PASTE MODEL FORMULA IN WORD DOC.


########## CONDITION STATS FOR MODIFY: 837, 856 ##########
data_modify <- subset(data_csv, (target_UID %in% modify_UIDs))
unique(data_modify$target_UID)

# Subset data
roi_of_interest = 'lang_LH_netw'
data_modify_roi <- subset(data_modify, (roi==roi_of_interest))

fname_save_modify = paste(DATAFNAME,'_',roi_of_interest,'_','modify.docx',sep="")


# Run LME
l.modify <- lmer(formula_full, data=data_modify_roi, REML=FALSE)
print(summary(l.modify))

modelsummary::modelsummary(l.modify, fmt=3, stars=TRUE, 
                           coef_rename = c("cond_factorD" = "condition_D", "cond_factorB" = "condition_S", "item_id_factor" = "sentence", "trial_num" = "trial_within_run"),
                           statistic = c("SE = {std.error}", "t = {statistic}","p = {p.value}", "df = {df.error}"), 
                           output = paste(SAVEDIR, SAVESUBDIR, fname_save_modify, sep=''))

# Run pairwise comparisons. PASTE THESE TO A COPY OF THE WORD DOC THAT WAS JUST GENERATED.
l.modify.res <- summary(pairs(emmeans(l.modify, "cond_factor", lmerTest.limit = 125000, pbkrtest.limit = 125000), reverse = FALSE)) # Get drive - suppress
l.modify.res.rev <- summary(pairs(emmeans(l.modify, "cond_factor", lmerTest.limit = 125000, pbkrtest.limit = 125000), reverse = TRUE)) # Get baseline

# PASTE MODEL FORMULA IN WORD DOC.








