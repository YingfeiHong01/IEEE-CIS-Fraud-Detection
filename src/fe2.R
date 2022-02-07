library(readr)
library(tidyverse)
library(MLmetrics)
library(data.table)
library(dplyr)
library(moments)

library(rpart)
library(rattle)
options(scipen = 99)

train_iden <- read_csv("train_identity.csv") %>% data.frame
train_trans <- read_csv("train_transaction.csv") %>% data.frame
test_iden <- read_csv("test_identity.csv") %>% data.frame
test_trans <- read_csv("test_transaction.csv") %>% data.frame

y <- train_trans$isFraud 
train_trans$isFraud <- NULL

drop_col <- c('V300','V309','V111','C3','V124','V106','V125','V315','V134','V102','V123','V316','V113',
              'V136','V305','V110','V299','V289','V286','V318','V103','V304','V116','V29','V284','V293',
              'V137','V295','V301','V104','V311','V115','V109','V119','V321','V114','V133','V122','V319',
              'V105','V112','V118','V117','V121','V108','V135','V320','V303','V297','V120')

train <- train_trans %>% left_join(train_iden)
test <- test_trans %>% left_join(test_iden)
train_rows = nrow(train)
rm(train_iden,train_trans,test_iden,test_trans) ; invisible(gc())

# using single hold-out validation (20%)
tr_idx <- which(train$TransactionDT < quantile(train$TransactionDT,0.66))
train[,drop_col] <- NULL
test[,drop_col] <- NULL


tem <- train %>% bind_rows(test) %>%
  mutate(hr = floor( (TransactionDT / 3600) %% 24 ),
         weekday = floor( (TransactionDT / 3600 / 24) %% 7)
  ) %>%
  select(-TransactionID,-TransactionDT)

#############################################################################################################
# FE part1 : Count encoding
tem$acc = paste(tem$card1,tem$card2,tem$card3,tem$card4,tem$card5,tem$addr1,tem$addr2)
tem$addr1_addr2 = paste(tem$addr1,tem$addr2)

char_features = c("acc","card1","card2","card3","ProductCD")

remove(train,test)

#############################################################################################################
# label 

char_features <- colnames(tem[, sapply(tem, class) %in% c('character', 'factor')])
for (f in char_features){
  levels <- unique(tem[[f]])
  tem[[f]] <- as.integer(factor(tem[[f]], levels=levels))
}

#FE part2: mean and sd transaction amount to card for card1, card4, id_02, D15
tem$decimal = nchar(tem$TransactionAmt - floor(tem$TransactionAmt))
fe_part2 <- tem[, c("acc","card1","card2","card3", "card4", "TransactionAmt", "id_02", "D15","addr1_addr2","decimal","C13","V258","C1","C14")]

fe_part2 <- fe_part2 %>% left_join(plyr::ddply(fe_part2, ~card1, summarise, mean_card1_Trans= mean(TransactionAmt), sd_card1_Trans = sd(TransactionAmt),count_card1 = n()))
fe_part2 <- fe_part2 %>% left_join(plyr::ddply(fe_part2, ~addr1_addr2, summarise, mean_geo_Trans= mean(TransactionAmt), sd_geo_Trans = sd(TransactionAmt),count_geo = n()))
fe_part2 <- fe_part2 %>% left_join(plyr::ddply(fe_part2, ~card1, summarise, min_card1_Trans= min(TransactionAmt), max_card1_Trans = max(TransactionAmt)))
fe_part2 <- fe_part2 %>% left_join(plyr::ddply(fe_part2, ~card2, summarise, min_card2_Trans= min(TransactionAmt), max_card2_Trans = max(TransactionAmt),count_card2 = n()))
fe_part2 <- fe_part2 %>% left_join(plyr::ddply(fe_part2, ~card4, summarise, mean_card4_Trans = mean(TransactionAmt), sd_card4_Trans = sd(TransactionAmt),count_card4 = n()))
fe_part2 <- fe_part2 %>% left_join(plyr::ddply(fe_part2, ~acc, summarise, mean_acc_Trans = mean(TransactionAmt), sd_acc_Trans = sd(TransactionAmt),count_acc = n()))



fe_part2 = fe_part2 %>% left_join(plyr::ddply(fe_part2, ~acc, summarise, mean_acc_dec = mean(decimal,na.rm = T), sd_acc_dec = sd(decimal),acc_dec_q = quantile(decimal,.8,na.rm = T)))

# %% [code]
head(fe_part2)
head(unique(fe_part2$mean_card1_D15))
head(unique(fe_part2$mean_card1_id02))
fe_part2 <- fe_part2[, -c(1:14)]

# do run length encoding on transaction amt?
# benefit of grouping together same valued transactions might pick up more rather than just using them in isolation.

char_features <- tem[,colnames(tem) %in% 
                       c("ProductCD","card1","card2","card3","card4","card5","card6","addr1","addr2","P_emaildomain",
                         "R_emaildomain","M1","M2","M3","M4","M5","M6","M7","M8","M9","DeviceType","DeviceInfo","id_12",
                         "id_13","id_14","id_15","id_16","id_17","id_18","id_19","id_20","id_21","id_22","id_23","id_24",
                         "id_25","id_26","id_27","id_28","id_29","id_30","id_31","id_32","id_33","id_34","id_35","id_36",
                         "id_37","id_38","acc")]

fe_part1 <- data.frame(0)
for(a in colnames(char_features) ){
  tem1 <- char_features %>% group_by(.dots = a) %>% mutate(count = length(card4)) %>% ungroup() %>% select(count)
  colnames(tem1) <- paste(a,"__count_encoding",sep="")
  fe_part1 <- data.frame(fe_part1,tem1)
}

fe_part1 <- fe_part1[,-1]
rm(char_features,tem1) ; invisible(gc())
cat("fe_part1 ncol :" , ncol(fe_part1) ,"\n" )
################################################

tem <- data.frame(tem,fe_part1,fe_part2)
remove(fe_part1,fe_part2,tem2)

tem = tem[!duplicated(as.list(tem))]
tem$addr1_card1 = paste(tem$addr1,tem$card1) 
tem$card1_and_count = paste(tem$card1,tem$count_card1)
tem$card2_and_count = paste(tem$card2,tem$count_card2)
tem$new3 = tem$TransactionAmt * tem$C1
tem$new5 = tem$TransactionAmt * tem$C13
tem$new7 = tem$TransactionAmt * tem$C14


#tem$lag_count = lag_counts
#tem$trans_diffs = trans_diffs

tem$NA_V1_V11 = apply(is.na(tem[,which(names(tem) %in% paste0("V",1:11))]), 1, sum)
tem$NA_V12_V34 = apply(is.na(tem[,which(names(tem) %in% paste0("V",12:34))]), 1, sum)
tem$NA_V35_V54 = apply(is.na(tem[,which(names(tem) %in% paste0("V",35:52))]), 1, sum)
tem$NA_V53_V74 = apply(is.na(tem[,which(names(tem) %in% paste0("V",53:74))]), 1, sum)
tem$NA_V75_V94 = apply(is.na(tem[,which(names(tem) %in% paste0("V",75:94))]), 1, sum)
tem$NA_V95_V137 = apply(is.na(tem[,which(names(tem) %in% paste0("V",95:137))]), 1, sum)
tem$NA_V138_V166 = apply(is.na(tem[,which(names(tem) %in% paste0("V",138:166))]), 1, sum)
tem$NA_V167_V216 = apply(is.na(tem[,which(names(tem) %in% paste0("V",167:216))]), 1, sum)
tem$NA_V217_V278 = apply(is.na(tem[,which(names(tem) %in% paste0("V",217:278))]), 1, sum)
tem$NA_V279_V321 = apply(is.na(tem[,which(names(tem) %in% paste0("V",279:321))]), 1, sum)
tem$NA_V322_V339 = apply(is.na(tem[,which(names(tem) %in% paste0("V",322:339))]), 1, sum)


remove(train_iden,train_trans,test_iden,test_trans,acc_info)
tem$decimal_diff = tem$decimal - tem$mean_acc_dec
tem$new13 = tem$decimal_diff * tem$mean_acc_Trans

#bad = fread("bad2.csv")


train <- tem[1:train_rows,]
test <- tem[-c(1:nrow(train)),]
remove(tem2)
# train$num_NA = apply(is.na(train[,1:358]), 1, sum)
# test$num_NA = apply(is.na(test[,1:358]), 1, sum)

#rm(tem) ; 
invisible(gc())

write.csv(train, file='train_r2.csv',row.names=FALSE)
write.csv(test, file='test_r2.csv',row.names=FALSE)