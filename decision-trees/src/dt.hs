import System.IO
import Control.Monad
import System.Environment
import Data.List
import Data.Set (toList, fromList)
import Data.Function (on)
import Control.Arrow (ArrowChoice(left))

import System.Random

data DS = DS [[Float]] [String] [String]

data Tree = Root Int Float Tree Tree | Leaf String

data GiniIndex =
    GiniIndex
        Float -- Total
        Float -- Left
        Float -- Right

instance Eq GiniIndex where
    (GiniIndex l1 l2 l3) == (GiniIndex r1 r2 r3) = l1 == r1 && l2 == r2 && l3 == r3

instance Ord GiniIndex where
    compare (GiniIndex i1 _ _) (GiniIndex i2 _ _) = i1 `compare` i2

data GiniSplit =
    GiniSplit
        Int       -- Attribute
        Float     -- Threshold
        GiniIndex -- Gini Index


instance Show Tree where
    show (Leaf value) = "{\"class\":\"" ++ value ++ "\"}"
    show (Root attr th l r) = "{\"attr\":" ++ show attr ++ ",\"th\":" ++ show th ++ ",\"l\":" ++ show l ++ ",\"r\":" ++ show r ++ "}"


treeDepth :: Tree -> Int
treeDepth (Leaf _) = 1
treeDepth (Root _ _ l r) = succ (max (treeDepth l) (treeDepth r))
    

genRandomMask :: Int -> [Bool]
genRandomMask len = generate baseGen len
    where
        dist = uniformR (1::Int, 5::Int)
        baseGen = mkStdGen 13
        generate gen len
            | len == 0  = []
            | otherwise = let
                (num, nextGen) = dist gen
            in
                (num == 5) : generate nextGen (pred len)

invertMask :: [Bool] -> [Bool]
invertMask = map not

filterByMask :: [a] -> [Bool] -> [a]
filterByMask lst mask = [value | (value, flag) <- zip lst mask, flag]

mode :: Ord a => [a] -> a
mode lst = maxVal
    where
        values = unique lst
        countValue val = (countOccurrences lst val, val)
        counts = map countValue values
        (count, maxVal) = maximumBy (compare `on` fst) counts


countOccurrences :: Eq a => [a] -> a -> Int
countOccurrences lst value = sum [1 :: Int | val <- lst, val == value]


predict :: Tree -> [Float] -> String
predict (Root attr split left right) record = correctTree `predict` record
    where
        desiredAttr = record !! attr
        correctTree
            | desiredAttr <= split = left
            | otherwise            = right

predict (Leaf value) record = value

numAttrs :: DS -> Int
numAttrs (DS (h: _) _ _) = length h


attrList :: DS -> [Int]
attrList ds = [0..(pred (numAttrs ds))]


unique :: Ord a => [a] -> [a]
unique = toList . fromList


nonEmpty :: [a] -> Bool
nonEmpty lst = not (null lst)


splitStr :: Char -> String -> [String]
splitStr delim str = filter nonEmpty (splitStr' delim str "")
    where
        splitStr' :: Char -> String -> String -> [String]
        splitStr' delim input current
            | null input          = [current]
            | head input == delim = current : splitStr' delim (tail input) ""
            | otherwise           = splitStr' delim (tail input) (current ++ [head input])


ticTacToeToFloat :: String -> Float
ticTacToeToFloat str
    | str == "x" = 0.0
    | str == "o" = 1.0
    | str == "b" = 2.0


floatToTicTacToe :: Float -> String
floatToTicTacToe num
    | num == 0.0 = "x"
    | num == 1.0 = "o"
    | num == 2.0 = "b"


readAttr :: String -> Float
readAttr = read


giniSteps = 5 :: Float


minimizeGini :: DS -> [Int] -> GiniSplit
minimizeGini ds attributes = 
    let
        getGiniIndex = giniOnAttr ds
        bestSplits   = map getGiniIndex attributes
        zipped       = zip bestSplits attributes
        -- Empty structure
        best         = minimumBy (compare `on` (snd . fst)) zipped

        ((threshold, giniIndex), attr) = best
    in
        GiniSplit attr threshold giniIndex


giniOnAttr :: DS -> Int -> (Float, GiniIndex)
giniOnAttr ds attr = let
        DS attrs classes classSet = ds

        desAttr   = map (!! attr) attrs
        minVal    = minimum desAttr
        maxVal    = maximum desAttr
        step      = (maxVal - minVal) / giniSteps
        range     = [minVal-step,minVal..maxVal+step]
        giniIdx   = giniForSplit desAttr classes classSet
        allSplits = [(split, giniIdx split) | split <- range ]
        bestSplit = minimumBy compare' allSplits

        compare' x y  = snd x `compare` snd y
    in
        bestSplit


giniForSplit :: [Float] -> [String] -> [String] -> Float -> GiniIndex
giniForSplit values classes classSet split =
    let 
        ziped    = zip values classes
        left     = filter ((<= split) . fst) ziped
        right    = filter ((>  split) . fst) ziped
        leftLen  = fromIntegral (length left)
        rightLen = fromIntegral (length right)
        total    = fromIntegral (length values)

        countClass :: [(Float, String)] -> String -> Int
        countClass lst cls = sum (map (\x -> if snd x == cls then 1 else 0) lst)

        leftCounts  = map (countClass left) classSet
        rightCounts = map (countClass right) classSet

        prob :: Float -> Int -> Float
        prob 0 _ = 0
        prob _ 0 = 0
        prob len count = (fromIntegral count / len)^2

        leftProbs  = map (prob leftLen) leftCounts
        rightProbs = map (prob rightLen) rightCounts

        giniLeft  = 1 - sum leftProbs
        giniRight = 1 - sum rightProbs

        giniTotal = giniLeft * (leftLen / total) + giniRight * (rightLen / total)
    in
        GiniIndex giniTotal giniLeft giniRight


createTree :: DS -> [Int] -> Int -> Tree
createTree (DS _ classes _) trainAttrs 1 = Leaf (mode classes)

createTree ds trainAttrs depth =
    let 
        DS attrs classes uniqueClasses        = ds
        GiniSplit splitOn threshold giniIndex = minimizeGini ds trainAttrs
        GiniIndex totalIdx leftIdx rightIdx   = giniIndex

        splitAttr      = map (!! splitOn) attrs
        zipped         = zip attrs classes

        leftMask       = map (<= threshold) splitAttr
        rightMask      = invertMask leftMask

        attributesLeft = filter (/= splitOn) trainAttrs

        -- If we got a prefect split, we can set depth to one to force
        -- the program to create leaves and prevent the creation of an
        -- unnecessarily deep tree
        leftDepth      = if leftIdx == 0 then 1 else pred depth
        rightDepth     = if rightIdx == 0 then 1 else pred depth

        valuesLeft     = filterByMask attrs leftMask
        classesLeft    = filterByMask classes leftMask
        leftTree       = createTree (DS valuesLeft classesLeft uniqueClasses) attributesLeft leftDepth

        valuesRight    = filterByMask attrs rightMask
        classesRight   = filterByMask classes rightMask
        rightTree      = createTree (DS valuesRight classesRight uniqueClasses) attributesLeft rightDepth

        tree           = Root splitOn threshold leftTree rightTree
        leaf           = Leaf (mode classes)

        result         = if totalIdx == 0 then leaf else tree
    in
        result


-- data DS = DS [[Float]] [String] [String]
testPrediction :: Tree -> DS -> Float
testPrediction tree (DS attrs classes _) = let
        predictions = map (predict tree) attrs
        isCorrect   = [pred == act | (pred, act) <- zip predictions classes]
        total       = fromIntegral (length classes)
        accurate    = fromIntegral (countOccurrences isCorrect True)
    in
        accurate / total


testDepth :: DS -> DS -> Int -> (Float, Float)
testDepth train test depth =
    let
        attrIndices   = attrList train
        tree          = createTree train attrIndices depth
        trainAcc      = testPrediction tree train
        testAcc       = testPrediction tree test
    in
        (trainAcc, testAcc)

testAll :: DS -> DS -> [(Float, Float)]
testAll train test =
    let
        depths  = length (attrList train)
        range   = [1..depths]
        results = map (testDepth train test) range
    in
        results

main = do
    args <- getArgs

    let infile    = head args
        depth     = read (args !! 1 ) :: Int
        header    = (args !! 2) `elem` ["y", "yes", "Y", "YES", "Yes", "1"]
        dataType  = args !! 3
        separator = head (args !! 4)

    contents <- readFile infile

    let allLines      = lines contents
        nonCR         = map (filter (/= '\r')) allLines
        nonEmpty      = filter (not . null) nonCR
        untyped       = if header then tail nonEmpty else nonEmpty
        split         = map (splitStr separator) untyped
        attrCols      = map init split
        classes       = map last split
        uniqueClasses = unique classes
        attributes    = map (map (if dataType == "char" then ticTacToeToFloat else readAttr)) attrCols
        trainMask     = genRandomMask (length attributes)
        validMask     = invertMask trainMask
        trainAttrs    = filterByMask attributes trainMask
        trainClasses  = filterByMask classes trainMask
        validAttrs    = filterByMask attributes validMask
        validClasses  = filterByMask classes validMask

        trainDf       = DS trainAttrs trainClasses uniqueClasses
        testDf        = DS validAttrs validClasses uniqueClasses
        results       = testAll trainDf testDf
        depths        = [1..(length (attrList trainDf))]
        zippedRes     = zip depths results
        strRes        = map resToStr zippedRes

        tree = createTree trainDf (attrList trainDf) depth

        resToStr (dep, acc) = show dep ++ ": " ++ show acc

    -- print trainAttrs
    -- print trainClasses
    print (attrList trainDf)
    print (treeDepth tree)
    print tree

    mapM putStrLn strRes
