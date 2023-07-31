library(tidyverse)
library(lme4)
library(lmerTest)
library(MuMIn)
library(emmeans)
library(modelsummary)
library(flextable)

##### SETTINGS #####
DATADIR = '/Users/gt/Documents/GitHub/drive_suppress_brains/data/'
DATAFNAME = 'brain-lang-data_participant_20230728.csv'
SAVEDIR = '/Users/gt/Documents/GitHub/drive_suppress_brains/src/statistics/linguistic_properties_stats/' # Define the directory where stats will be saved
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

roi = 'lang_LH_netw'
search_UIDs <- c('797', '841', '880')


#### FUNCTION FOR RUNNING LME ####
run_lme <- function(data,
                    roi_of_interest,
                    formula,
                    REML) {
  
  # Subset data
  data_roi <- subset(data, (roi==roi_of_interest))
  
  print("Number of rows: ")
  print(nrow(data_roi))
  
  # Run LME
  lme_model <- lmer(formula, data=data_roi, REML=REML)
  print(summary(lme_model))
  
  return(lme_model)
}

##### LOAD DATA #####
data_csv = read.csv(paste(DATADIR, DATAFNAME, sep=''), header=TRUE)

length(unique(data_csv$target_UID))
length(unique(data_csv$roi))

# Factorize variables
# Convert item_id into a factor because we do not care which item_id is which 
data_csv$item_id_factor <- factor(data_csv$item_id)


########## BASELINE SET ##########
data_B <- subset(data_csv, (cond %in% c("B")))

## Base LME (only surprisal)
l = run_lme(data = data_B,
            roi_of_interest = 'lang_LH_netw',
            formula = response_target ~ log.prob.gpt2.xl_mean + (1 | item_id_factor) + run_within_session + trial_num, 
            REML=FALSE)

modelsummary::modelsummary(l, fmt=3, stars=TRUE, 
                           coef_rename = c("log.prob.gpt2.xl_mean" = "log_probability", "item_id_factor" = "sentence", "trial_num" = "trial_within_run"),
                           statistic = c("SE = {std.error}", "t = {statistic}","p = {p.value}", "df = {df.error}"), 
                           output = paste(SAVEDIR, SAVESUBDIR, DATAFNAME, '_preds=log-prob.docx', sep=''))


## Compare all variables!
feats_above_surp = c('rating_arousal_mean',
                     'rating_conversational_mean', 
                     'rating_sense_mean', 
                     'rating_gram_mean',
                     'rating_imageability_mean', 
                     'rating_others_thoughts_mean', 
                     'rating_physical_mean',
                     'rating_places_mean',
                     'rating_valence_mean')

## 1. Compare surprisal, sense, gram bucket
l_sense = run_lme(data = data_B,
            roi_of_interest = 'lang_LH_netw',
            formula = response_target ~ log.prob.gpt2.xl_mean + rating_sense_mean + (1 | item_id_factor) + run_within_session + trial_num, 
            REML=FALSE)
modelsummary::modelsummary(l_sense, fmt=3, stars=TRUE, 
                           coef_rename = c("log.prob.gpt2.xl_mean" = "log_probability", "rating_sense_mean" = "plausibility", "item_id_factor" = "sentence", "trial_num" = "trial_within_run"),
                           statistic = c("SE = {std.error}", "t = {statistic}","p = {p.value}", "df = {df.error}"), 
                           output = paste(SAVEDIR, SAVESUBDIR, DATAFNAME, '_preds=log-prob_plaus.docx', sep=''))


l_sense_gram = run_lme(data = data_B,
                  roi_of_interest = 'lang_LH_netw',
                  formula = response_target ~ log.prob.gpt2.xl_mean + rating_sense_mean + rating_gram_mean + (1 | item_id_factor) + run_within_session + trial_num, 
                  REML=FALSE)
modelsummary::modelsummary(l_sense_gram, fmt=3, stars=TRUE, 
                           coef_rename = c("log.prob.gpt2.xl_mean" = "log_probability", "rating_sense_mean" = "plausibility", "rating_gram_mean" = "grammaticality", "item_id_factor" = "sentence", "trial_num" = "trial_within_run"),
                           statistic = c("SE = {std.error}", "t = {statistic}","p = {p.value}", "df = {df.error}"), 
                           output = paste(SAVEDIR, SAVESUBDIR, DATAFNAME, '_preds=log-prob_plaus_gram.docx', sep=''))

t_sense_sense_gram <- lme4:::anova.merMod(l_sense, l_sense_gram)
save_as_docx(flextable(t_sense_sense_gram), path=paste(SAVEDIR, SAVESUBDIR, 'anova=sense_vs_sense-gram_', DATAFNAME, '.docx', sep=''))


l_gram = run_lme(data = data_B,
                 roi_of_interest = 'lang_LH_netw',
                 formula = response_target ~ log.prob.gpt2.xl_mean + rating_gram_mean + (1 | item_id_factor) + run_within_session + trial_num, 
                 REML=FALSE)
modelsummary::modelsummary(l_gram, fmt=3, stars=TRUE, 
                           coef_rename = c("log.prob.gpt2.xl_mean" = "log_probability", "rating_sense_mean" = "plausibility", "item_id_factor" = "sentence", "trial_num" = "trial_within_run"),
                           statistic = c("SE = {std.error}", "t = {statistic}","p = {p.value}", "df = {df.error}"), 
                           output = paste(SAVEDIR, SAVESUBDIR, DATAFNAME, '_preds=log-prob_gram.docx', sep=''))

l_gram_sense = run_lme(data = data_B,
                 roi_of_interest = 'lang_LH_netw',
                 formula = response_target ~ log.prob.gpt2.xl_mean + rating_gram_mean + rating_sense_mean + (1 | item_id_factor) + run_within_session + trial_num, 
                 REML=FALSE) # identical to sense_gram, order does not matter

t_gram_sense_gram <- lme4:::anova.merMod(l_gram, l_gram_sense)
save_as_docx(flextable(t_gram_sense_gram), path=paste(SAVEDIR, SAVESUBDIR, 'anova=gram_vs_sense-gram_', DATAFNAME, '.docx', sep=''))


## 2. Compare content bucket: others thoughts, physical, places
# No effect of others_thoughts beyond surp
l_mental = run_lme(data = data_B,
                 roi_of_interest = 'lang_LH_netw',
                 formula = response_target ~ log.prob.gpt2.xl_mean + rating_others_thoughts_mean + (1 | item_id_factor) + run_within_session + trial_num, 
                 REML=FALSE)
modelsummary::modelsummary(l_mental, fmt=3, stars=TRUE, 
                           coef_rename = c("log.prob.gpt2.xl_mean" = "log_probability", "rating_others_thoughts_mean" = "mental_states", "item_id_factor" = "sentence", "trial_num" = "trial_within_run"),
                           statistic = c("SE = {std.error}", "t = {statistic}","p = {p.value}", "df = {df.error}"), 
                           output = paste(SAVEDIR, SAVESUBDIR, DATAFNAME, '_preds=log-prob_mental.docx', sep=''))

t_mental <- lme4:::anova.merMod(l, l_mental)
save_as_docx(flextable(t_mental), path=paste(SAVEDIR, SAVESUBDIR, 'anova=prob_vs_mental_', DATAFNAME, '.docx', sep=''))


l_phys = run_lme(data = data_B,
                  roi_of_interest = 'lang_LH_netw',
                  formula = response_target ~ log.prob.gpt2.xl_mean + rating_physical_mean + (1 | item_id_factor) + run_within_session + trial_num, 
                  REML=FALSE)
modelsummary::modelsummary(l_phys, fmt=3, stars=TRUE, 
                           coef_rename = c("log.prob.gpt2.xl_mean" = "log_probability", "rating_physical_mean" = "physical_objects", "item_id_factor" = "sentence", "trial_num" = "trial_within_run"),
                           statistic = c("SE = {std.error}", "t = {statistic}","p = {p.value}", "df = {df.error}"), 
                           output = paste(SAVEDIR, SAVESUBDIR, DATAFNAME, '_preds=log-prob_phys.docx', sep=''))

t_phys <- lme4:::anova.merMod(l, l_phys)
save_as_docx(flextable(t_phys), path=paste(SAVEDIR, SAVESUBDIR, 'anova=prob_vs_phys_', DATAFNAME, '.docx', sep=''))


l_places = run_lme(data = data_B,
                 roi_of_interest = 'lang_LH_netw',
                 formula = response_target ~ log.prob.gpt2.xl_mean + rating_places_mean + (1 | item_id_factor) + run_within_session + trial_num, 
                 REML=FALSE)
modelsummary::modelsummary(l_places, fmt=3, stars=TRUE, 
                           coef_rename = c("log.prob.gpt2.xl_mean" = "log_probability", "rating_places_mean" = "places", "item_id_factor" = "sentence", "trial_num" = "trial_within_run"),
                           statistic = c("SE = {std.error}", "t = {statistic}","p = {p.value}", "df = {df.error}"), 
                           output = paste(SAVEDIR, SAVESUBDIR, DATAFNAME, '_preds=log-prob_places.docx', sep=''))

t_places <- lme4:::anova.merMod(l, l_places)
save_as_docx(flextable(t_places), path=paste(SAVEDIR, SAVESUBDIR, 'anova=prob_vs_places_', DATAFNAME, '.docx', sep=''))



## 3. Compare emotion bucket: valence, arousal
l_val = run_lme(data = data_B,
                 roi_of_interest = 'lang_LH_netw',
                 formula = response_target ~ log.prob.gpt2.xl_mean + rating_valence_mean + (1 | item_id_factor) + run_within_session + trial_num, 
                 REML=FALSE)
modelsummary::modelsummary(l_val, fmt=3, stars=TRUE, 
                           coef_rename = c("log.prob.gpt2.xl_mean" = "log_probability", "rating_valence_mean" = "valence", "item_id_factor" = "sentence", "trial_num" = "trial_within_run"),
                           statistic = c("SE = {std.error}", "t = {statistic}","p = {p.value}", "df = {df.error}"), 
                           output = paste(SAVEDIR, SAVESUBDIR, DATAFNAME, '_preds=log-prob_valence.docx', sep=''))

t_val <- lme4:::anova.merMod(l, l_val)
save_as_docx(flextable(t_val), path=paste(SAVEDIR, SAVESUBDIR, 'anova=prob_vs_val_', DATAFNAME, '.docx', sep=''))



l_arousal = run_lme(data = data_B,
                roi_of_interest = 'lang_LH_netw',
                formula = response_target ~ log.prob.gpt2.xl_mean + rating_arousal_mean + (1 | item_id_factor) + run_within_session + trial_num, 
                REML=FALSE)
modelsummary::modelsummary(l_arousal, fmt=3, stars=TRUE, 
                           coef_rename = c("log.prob.gpt2.xl_mean" = "log_probability", "rating_arousal_mean" = "arousal", "item_id_factor" = "sentence", "trial_num" = "trial_within_run"),
                           statistic = c("SE = {std.error}", "t = {statistic}","p = {p.value}", "df = {df.error}"), 
                           output = paste(SAVEDIR, SAVESUBDIR, DATAFNAME, '_preds=log-prob_arousal.docx', sep=''))

t_arousal <- lme4:::anova.merMod(l, l_arousal)
save_as_docx(flextable(t_arousal), path=paste(SAVEDIR, SAVESUBDIR, 'anova=prob_vs_arousal_', DATAFNAME, '.docx', sep=''))


## 4. Compare imageability bucket 
l_img = run_lme(data = data_B,
                   roi_of_interest = 'lang_LH_netw',
                   formula = response_target ~ log.prob.gpt2.xl_mean + rating_imageability_mean + (1 | item_id_factor) + run_within_session + trial_num, 
                   REML=FALSE)
modelsummary::modelsummary(l_img, fmt=3, stars=TRUE, 
                           coef_rename = c("log.prob.gpt2.xl_mean" = "log_probability", "rating_imageability_mean" = "imageability", "item_id_factor" = "sentence", "trial_num" = "trial_within_run"),
                           statistic = c("SE = {std.error}", "t = {statistic}","p = {p.value}", "df = {df.error}"), 
                           output = paste(SAVEDIR, SAVESUBDIR, DATAFNAME, '_preds=log-prob_img.docx', sep=''))

t_img <- lme4:::anova.merMod(l, l_img)
save_as_docx(flextable(t_img), path=paste(SAVEDIR, SAVESUBDIR, 'anova=prob_vs_img_', DATAFNAME, '.docx', sep=''))


## 5. Compare frequency bucket 
l_freq = run_lme(data = data_B,
                roi_of_interest = 'lang_LH_netw',
                formula = response_target ~ log.prob.gpt2.xl_mean + rating_frequency_mean + (1 | item_id_factor) + run_within_session + trial_num, 
                REML=FALSE)
modelsummary::modelsummary(l_freq, fmt=3, stars=TRUE, 
                           coef_rename = c("log.prob.gpt2.xl_mean" = "log_probability", "rating_frequency_mean" = "general_frequency", "item_id_factor" = "sentence", "trial_num" = "trial_within_run"),
                           statistic = c("SE = {std.error}", "t = {statistic}","p = {p.value}", "df = {df.error}"), 
                           output = paste(SAVEDIR, SAVESUBDIR, DATAFNAME, '_preds=log-prob_freq.docx', sep=''))

t_freq <- lme4:::anova.merMod(l, l_freq)
save_as_docx(flextable(t_freq), path=paste(SAVEDIR, SAVESUBDIR, 'anova=prob_vs_freq_', DATAFNAME, '.docx', sep=''))



l_conv= run_lme(data = data_B,
                 roi_of_interest = 'lang_LH_netw',
                 formula = response_target ~ log.prob.gpt2.xl_mean + rating_conversational_mean + (1 | item_id_factor) + run_within_session + trial_num, 
                 REML=FALSE)
modelsummary::modelsummary(l_conv, fmt=3, stars=TRUE, 
                           coef_rename = c("log.prob.gpt2.xl_mean" = "log_probability", "rating_conversational_mean" = "conversational_frequency", "item_id_factor" = "sentence", "trial_num" = "trial_within_run"),
                           statistic = c("SE = {std.error}", "t = {statistic}","p = {p.value}", "df = {df.error}"), 
                           output = paste(SAVEDIR, SAVESUBDIR, DATAFNAME, '_preds=log-prob_conv.docx', sep=''))

t_conv <- lme4:::anova.merMod(l, l_conv)
save_as_docx(flextable(t_conv), path=paste(SAVEDIR, SAVESUBDIR, 'anova=prob_vs_conv_', DATAFNAME, '.docx', sep=''))



