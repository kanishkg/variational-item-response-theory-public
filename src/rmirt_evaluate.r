library("mirt")
library("mirtCAT")
library("reticulate")
library("gsubfn")

np <- import("numpy")

artificially_mask_dataset <- function(ratio, response, mask){
    attempted <- which(!is.na(response), arr.ind=TRUE)
    num_attempted <- nrow(attempted)
    num_to_impute <- round(ratio * num_attempted)
    impute_idx <- sample(1:num_to_impute, num_to_impute, replace = FALSE)
    missing_indices = data.frame(matrix(NA, nrow = num_to_impute, ncol = 2))
    missing_labels <- rep(NA, num_to_impute)
    for (i in 1:num_to_impute){
        idx <- attempted[impute_idx[i], 1:2]
        missing_labels[i] <- response[idx[1], idx[2]]
        missing_indices[i, 1:2] <- idx
        response[idx[1], idx[2]] <- NA
        mask[idx[1], idx[2]] <- 0 
    } 
    return (list(response, mask, missing_indices, missing_labels))
}

mask_encoder <- function(n, response) {
    attempted <- which(!is.na(response), arr.ind=TRUE)
    num_attempted <- nrow(attempted)
    num_to_impute <- num_attempted - n
    impute_idx <- sample(1:num_to_impute, num_to_impute, replace = FALSE)

    for (i in 1:num_to_impute){
        idx <- attempted[impute_idx[i], 1:2]
        response[idx[1], idx[2]] <- NA
    } 
    return (response)
}

getROC_AUC = function(probs, true_Y){
    probsSort = sort(probs, decreasing = TRUE, index.return = TRUE)
    val = unlist(probsSort$x)
    idx = unlist(probsSort$ix)  

    roc_y = true_Y[idx];
    stack_x = cumsum(roc_y == 2)/sum(roc_y == 2)
    stack_y = cumsum(roc_y == 1)/sum(roc_y == 1)    

    auc = sum((stack_x[2:length(roc_y)]-stack_x[1:length(roc_y)-1])*stack_y[2:length(roc_y)])
    return(auc)
}

predict_response <- function(ability, pars){
    # $$P(x = 1|\theta, \psi) = g + \frac{(u - g)}{ 1 + exp(-(a_1 * \theta_1 + a_2 * \theta_2 + d))}$$
    a = pars[1]
    d = pars[2]
    g = pars[3]
    u = pars[4]
    theta = ability
    logit <- a*theta + d
    pred <- g + (u-g/(1+exp(-logit)))
    return (pred)
}

predict <- function(missing_indices, item_coeffs, predicted_ability){
    predicted_labels <- rep(NA, nrow(missing_indices))
    for (i in 1:nrow(missing_indices)){
        print(paste("missing indices: ", missing_indices[i, 1:2]))
        idx <- missing_indices[i, 1:2]
        print(paste("idx: ", idx))
        print(paste("predicted_ab: ", predicted_ability))
        ability <- predicted_ability[idx[1], 1]
        pars <- item_coeffs[idx[2]]
        pars <- pars[[names(pars)[1]]]
        predicted_response = predict_response(ability, pars)
        predicted_labels[i] <- predicted_response
    }
    return (predicted_labels)
}

# read files
machine_response <- np$load("data/r_irt/algebraai_response.npy")
machine_response <- as.data.frame(machine_response)
machine_mask <- np$load("data/r_irt/algebraai_mask.npy")
machine_mask <- as.data.frame(machine_mask)
human_response <- np$load("data/r_irt/human_response.npy")
human_response <- as.data.frame(human_response)
human_mask <- np$load("data/r_irt/human_mask.npy")
human_mask <- as.data.frame(human_mask)
# mask responses
for (i in 1:nrow(human_response)) {
    for (j in 1:ncol(human_response)) {
        if (human_mask[i, j] == 0) {
            human_response[i, j] <- NA
        }
    }
} 
num_correct <- rowSums(human_response, na.rm=TRUE)
num_attempted <- rowSums(human_mask, na.rm=TRUE)
empirical_ability <- num_correct / num_attempted

irt_params <- mirt(data = machine_response, model = 1, itemtype='2PL')
item_coeffs <- coef(irt_params)


for (n in 2:11){
    for (s in 1:20){
        set.seed(s)
        # impute dataset
        list[imputed_response, imputed_mask, missing_indices, missing_labels] <- artificially_mask_dataset(0.1, human_response, human_mask)
        # mask encoder
        if (n != 11) {
            masked_response <- mask_encoder(n, imputed_response)
        } else {
            masked_response <- imputed_response
        }
        # get predicted ability
        predicted_ability <- fscores(irt_params, response.pattern = masked_response)
        # get correlation
        corr <- cor(predicted_ability[1:nrow(predicted_ability), 1], empirical_ability, method = "pearson")
        print(paste("corr: ", corr))
        # predict imputed responses
        predicted_response <- predict(missing_indices, item_coeffs, predicted_ability)
        # compute metrics
        cm <- as.matrix(table(Actual = missing_labels, Predicted = predicted_response))
        accuracy <- sum(diag(cm)) / sum(cm)
        auc <- getROC_AUC(predicted_response, missing_labels)
        print(paste("n = ", n, ", s = ", s, ", corr = ", corr, ", accuracy = ", accuracy, ", auc = ", auc))
    }
}