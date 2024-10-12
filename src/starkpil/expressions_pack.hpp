#ifndef EXPRESSIONS_PACK_HPP
#define EXPRESSIONS_PACK_HPP
#include "expressions_ctx.hpp"

class ExpressionsPack : public ExpressionsCtx {
public:
    uint64_t nrowsPack = 4;
    uint64_t nCols;
    vector<uint64_t> nColsStages;
    vector<uint64_t> nColsStagesAcc;
    vector<uint64_t> offsetsStages;
    ExpressionsPack(SetupCtx& setupCtx) : ExpressionsCtx(setupCtx) {};

    void setBufferTInfo(bool domainExtended, uint64_t expId) {
        uint64_t nOpenings = setupCtx.starkInfo.openingPoints.size();
        offsetsStages.resize((setupCtx.starkInfo.nStages + 2)*nOpenings + 1);
        nColsStages.resize((setupCtx.starkInfo.nStages + 2)*nOpenings + 1);
        nColsStagesAcc.resize((setupCtx.starkInfo.nStages + 2)*nOpenings + 1);

        nCols = setupCtx.starkInfo.nConstants;
        uint64_t ns = setupCtx.starkInfo.nStages + 2;
        for(uint64_t o = 0; o < nOpenings; ++o) {
            for(uint64_t stage = 0; stage <= ns; ++stage) {
                std::string section = stage == 0 ? "const" : "cm" + to_string(stage);
                offsetsStages[(setupCtx.starkInfo.nStages + 2)*o + stage] = setupCtx.starkInfo.mapOffsets[std::make_pair(section, domainExtended)];
                nColsStages[(setupCtx.starkInfo.nStages + 2)*o + stage] = setupCtx.starkInfo.mapSectionsN[section];
                nColsStagesAcc[(setupCtx.starkInfo.nStages + 2)*o + stage] = stage == 0 && o == 0 ? 0 : nColsStagesAcc[(setupCtx.starkInfo.nStages + 2)*o + stage - 1] + nColsStages[stage - 1];
            }
        }

        nColsStagesAcc[(setupCtx.starkInfo.nStages + 2)*nOpenings] = nColsStagesAcc[(setupCtx.starkInfo.nStages + 2)*nOpenings - 1] + nColsStages[(setupCtx.starkInfo.nStages + 2)*nOpenings - 1];
        if(expId == setupCtx.starkInfo.cExpId) {
            nCols = nColsStagesAcc[(setupCtx.starkInfo.nStages + 2)*nOpenings] + setupCtx.starkInfo.boundaries.size() + 1;
        } else if(expId == setupCtx.starkInfo.friExpId) {
            nCols = nColsStagesAcc[(setupCtx.starkInfo.nStages + 2)*nOpenings] + nOpenings*FIELD_EXTENSION;
        } else {
            nCols = nColsStagesAcc[(setupCtx.starkInfo.nStages + 2)*nOpenings] + 1;
        }
    }

    inline void loadPolynomials(StepsParams& params, ParserArgs &parserArgs, std::vector<Dest> &dests, Goldilocks::Element *bufferT_, uint64_t row, bool domainExtended) {
        uint64_t nOpenings = setupCtx.starkInfo.openingPoints.size();
        uint64_t domainSize = domainExtended ? 1 << setupCtx.starkInfo.starkStruct.nBitsExt : 1 << setupCtx.starkInfo.starkStruct.nBits;

        uint64_t extendBits = (setupCtx.starkInfo.starkStruct.nBitsExt - setupCtx.starkInfo.starkStruct.nBits);
        int64_t extend = domainExtended ? (1 << extendBits) : 1;
        uint64_t nextStrides[nOpenings];
        for(uint64_t i = 0; i < nOpenings; ++i) {
            uint64_t opening = setupCtx.starkInfo.openingPoints[i] < 0 ? setupCtx.starkInfo.openingPoints[i] + domainSize : setupCtx.starkInfo.openingPoints[i];
            nextStrides[i] = opening * extend;
        }

        Goldilocks::Element *constPols = domainExtended ? setupCtx.constPols.pConstPolsAddressExtended : setupCtx.constPols.pConstPolsAddress;

        std::vector<bool> constPolsUsed(setupCtx.starkInfo.constPolsMap.size(), false);
        std::vector<bool> cmPolsUsed(setupCtx.starkInfo.cmPolsMap.size(), false);

        for(uint64_t i = 0; i < dests.size(); ++i) {
            uint16_t* cmUsed = &parserArgs.cmPolsIds[dests[i].params[0].parserParams.cmPolsOffset];
            uint16_t* constUsed = &parserArgs.constPolsIds[dests[i].params[0].parserParams.constPolsOffset];

            for(uint64_t k = 0; k < dests[i].params[0].parserParams.nConstPolsUsed; ++k) {
                constPolsUsed[constUsed[k]] = true;
            }

            for(uint64_t k = 0; k < dests[i].params[0].parserParams.nCmPolsUsed; ++k) {
                cmPolsUsed[cmUsed[k]] = true;
            }
        }
        for(uint64_t k = 0; k < constPolsUsed.size(); ++k) {
            if(!constPolsUsed[k]) continue;
            for(uint64_t o = 0; o < nOpenings; ++o) {
                for(uint64_t j = 0; j < nrowsPack; ++j) {
                    uint64_t l = (row + j + nextStrides[o]) % domainSize;
                    bufferT_[(nColsStagesAcc[(setupCtx.starkInfo.nStages + 2)*o] + k)*nrowsPack + j] = constPols[l * nColsStages[0] + k];
                }
            }
        }

        for(uint64_t k = 0; k < cmPolsUsed.size(); ++k) {
            if(!cmPolsUsed[k]) continue;
            PolMap polInfo = setupCtx.starkInfo.cmPolsMap[k];
            uint64_t stage = polInfo.stage;
            uint64_t stagePos = polInfo.stagePos;
            for(uint64_t d = 0; d < polInfo.dim; ++d) {
                for(uint64_t o = 0; o < nOpenings; ++o) {
                    for(uint64_t j = 0; j < nrowsPack; ++j) {
                        uint64_t l = (row + j + nextStrides[o]) % domainSize;
                        bufferT_[(nColsStagesAcc[(setupCtx.starkInfo.nStages + 2)*o + stage] + (stagePos + d))*nrowsPack + j] = params.pols[offsetsStages[stage] + l * nColsStages[stage] + stagePos + d];
                    }
                }
            }
        }

        if(dests[0].params[0].parserParams.expId == setupCtx.starkInfo.cExpId) {
            for(uint64_t d = 0; d < setupCtx.starkInfo.boundaries.size(); ++d) {
                for(uint64_t j = 0; j < nrowsPack; ++j) {
                    bufferT_[(nColsStagesAcc[(setupCtx.starkInfo.nStages + 2)*nOpenings] + d + 1)*nrowsPack + j] = setupCtx.constPols.zi[row + j + d*domainSize];
                }
            }
            for(uint64_t j = 0; j < nrowsPack; ++j) {
                bufferT_[(nColsStagesAcc[(setupCtx.starkInfo.nStages + 2)*nOpenings])*nrowsPack + j] = setupCtx.constPols.x_2ns[row + j];
            }
        } else if(dests[0].params[0].parserParams.expId == setupCtx.starkInfo.friExpId) {
            for(uint64_t d = 0; d < setupCtx.starkInfo.openingPoints.size(); ++d) {
               for(uint64_t k = 0; k < FIELD_EXTENSION; ++k) {
                    for(uint64_t j = 0; j < nrowsPack; ++j) {
                        bufferT_[(nColsStagesAcc[(setupCtx.starkInfo.nStages + 2)*nOpenings] + d*FIELD_EXTENSION + k)*nrowsPack + j] = params.xDivXSub[(row + j + d*domainSize)*FIELD_EXTENSION + k];
                    }
                }
            }
        } else {
            for(uint64_t j = 0; j < nrowsPack; ++j) {
                bufferT_[(nColsStagesAcc[(setupCtx.starkInfo.nStages + 2)*nOpenings])*nrowsPack + j] = setupCtx.constPols.x[row + j];
            }
        }
    }

    inline void copyPolynomial(Goldilocks::Element* destVals, bool inverse, uint64_t destId, uint64_t destDim, Goldilocks::Element* tmp1, Goldilocks::Element* tmp3) {
        // if(destDim == 1) {
        //     if(inverse) {
        //         Goldilocks::Element buff[nrowsPack];
        //         Goldilocks::store_avx(buff, tmp1[destId]);
        //         // for(uint64_t j = 0; j < nrowsPack; ++j) {
        //         //     Goldilocks::inv(dests[i].dest[(row + j)*offset], dests[i].dest[(row + j)*offset]);
        //         // }
        //         Goldilocks::batchInverse(buff, buff, nrowsPack);
        //         Goldilocks::load_avx(destVals[0], buff);
        //     } else {
        //         Goldilocks::copy_pack(nrowsPack, &destVals[0],&tmp1[destId*nrowsPack]);
        //     }
        // } else if(destDim == 3) {
        //     if(inverse) {
        //         Goldilocks::Element buff[FIELD_EXTENSION*nrowsPack];
        //         Goldilocks::store_avx(&buff[0], uint64_t(FIELD_EXTENSION), tmp3[destId][0]);
        //         Goldilocks::store_avx(&buff[1], uint64_t(FIELD_EXTENSION), tmp3[destId][1]);
        //         Goldilocks::store_avx(&buff[2], uint64_t(FIELD_EXTENSION), tmp3[destId][2]);
        //         Goldilocks3::batchInverse((Goldilocks3::Element *)buff, (Goldilocks3::Element *)buff, nrowsPack);
        //         Goldilocks::load_avx(destVals[0], &buff[0], uint64_t(FIELD_EXTENSION));
        //         Goldilocks::load_avx(destVals[1], &buff[1], uint64_t(FIELD_EXTENSION));
        //         Goldilocks::load_avx(destVals[2], &buff[2], uint64_t(FIELD_EXTENSION));
        //     } else {
        //         Goldilocks::copy_pack(nrowsPack, &destVals[0], &tmp3[destId*FIELD_EXTENSION*nrowsPack]);
        //         Goldilocks::copy_pack(nrowsPack, &destVals[1],&tmp3[(destId*FIELD_EXTENSION + 1)*nrowsPack]);
        //         Goldilocks::copy_pack(nrowsPack, &destVals[2],&tmp3[(destId*FIELD_EXTENSION + 2)*nrowsPack]);
        //     }
        // }
    }

    inline void storePolynomial(std::vector<Dest> dests, Goldilocks::Element* destVals, uint64_t row) {
        for(uint64_t i = 0; i < dests.size(); ++i) {
            if(dests[i].params[0].parserParams.destDim == 1) {
                uint64_t offset = dests[i].params[0].offset != 0 ? dests[i].params[0].offset : 1;
                Goldilocks::copy_pack(nrowsPack, &dests[i].dest[row*offset], uint64_t(offset), &destVals[i*FIELD_EXTENSION*nrowsPack]);
            } else {
                uint64_t offset = dests[i].params[0].offset != 0 ? dests[i].params[0].offset : FIELD_EXTENSION;
                Goldilocks::copy_pack(nrowsPack, &dests[i].dest[row*offset], uint64_t(offset), &destVals[i*FIELD_EXTENSION*nrowsPack]);
                Goldilocks::copy_pack(nrowsPack, &dests[i].dest[row*offset + 1], uint64_t(offset), &destVals[(i*FIELD_EXTENSION + 1)*nrowsPack]);
                Goldilocks::copy_pack(nrowsPack, &dests[i].dest[row*offset + 2], uint64_t(offset), &destVals[(i*FIELD_EXTENSION + 2)*nrowsPack]);
            }
        }
    }

    inline void printTmp1(uint64_t row, Goldilocks::Element* tmp) {
        Goldilocks::Element buff[nrowsPack];
        Goldilocks::copy_pack(nrowsPack, buff, tmp);
        for(uint64_t i = 0; i < 1; ++i) {
            cout << "Value at row " << row + i << " is " << Goldilocks::toString(buff[i]) << endl;
        }
    }

    inline void printTmp3(uint64_t row, Goldilocks::Element* tmp) {
        Goldilocks::Element buff[FIELD_EXTENSION*nrowsPack];
        Goldilocks::copy_pack(nrowsPack, &buff[0], uint64_t(FIELD_EXTENSION), &tmp[0]);
        Goldilocks::copy_pack(nrowsPack, &buff[1], uint64_t(FIELD_EXTENSION), &tmp[1]);
        Goldilocks::copy_pack(nrowsPack, &buff[2], uint64_t(FIELD_EXTENSION), &tmp[2]);
        for(uint64_t i = 0; i < 1; ++i) {
            cout << "Value at row " << row + i << " is [" << Goldilocks::toString(buff[FIELD_EXTENSION*i]) << ", " << Goldilocks::toString(buff[FIELD_EXTENSION*i + 1]) << ", " << Goldilocks::toString(buff[FIELD_EXTENSION*i + 2]) << "]" << endl;
        }
    }

    inline void printCommit(uint64_t row, Goldilocks::Element* bufferT, bool extended) {
        if(extended) {
            Goldilocks::Element buff[FIELD_EXTENSION*nrowsPack];
            Goldilocks::copy_pack(nrowsPack, &buff[0], uint64_t(FIELD_EXTENSION), &bufferT[0]);
            Goldilocks::copy_pack(nrowsPack, &buff[1], uint64_t(FIELD_EXTENSION), &bufferT[setupCtx.starkInfo.openingPoints.size()]);
            Goldilocks::copy_pack(nrowsPack, &buff[2], uint64_t(FIELD_EXTENSION), &bufferT[2*setupCtx.starkInfo.openingPoints.size()]);
            for(uint64_t i = 0; i < 1; ++i) {
                cout << "Value at row " << row + i << " is [" << Goldilocks::toString(buff[FIELD_EXTENSION*i]) << ", " << Goldilocks::toString(buff[FIELD_EXTENSION*i + 1]) << ", " << Goldilocks::toString(buff[FIELD_EXTENSION*i + 2]) << "]" << endl;
            }
        } else {
            Goldilocks::Element buff[nrowsPack];
            Goldilocks::copy_pack(nrowsPack, &buff[0], &bufferT[0]);
            for(uint64_t i = 0; i < 1; ++i) {
                cout << "Value at row " << row + i << " is " << Goldilocks::toString(buff[i]) << endl;
            }
        }
    }

    void calculateExpressions(StepsParams& params, ParserArgs &parserArgs, std::vector<Dest> dests, bool domainExtended) override {
        uint64_t nOpenings = setupCtx.starkInfo.openingPoints.size();
        uint64_t domainSize = domainExtended ? 1 << setupCtx.starkInfo.starkStruct.nBitsExt : 1 << setupCtx.starkInfo.starkStruct.nBits;

        setBufferTInfo(domainExtended, dests[0].params[0].parserParams.expId);

        Goldilocks::Element challenges[setupCtx.starkInfo.challengesMap.size()*FIELD_EXTENSION*nrowsPack];
        for(uint64_t i = 0; i < setupCtx.starkInfo.challengesMap.size(); ++i) {
            for(uint64_t j = 0; j < nrowsPack; ++j) {
                challenges[(i*FIELD_EXTENSION)*nrowsPack + j] = params.challenges[i * FIELD_EXTENSION];
                challenges[(i*FIELD_EXTENSION + 1)*nrowsPack + j] = params.challenges[i * FIELD_EXTENSION + 1];
                challenges[(i*FIELD_EXTENSION + 2)*nrowsPack + j] = params.challenges[i * FIELD_EXTENSION + 2];
            }
        }

        Goldilocks::Element** numbers_ = new Goldilocks::Element*[dests.size()];
        for(uint64_t i = 0; i < dests.size(); ++i) {
            uint64_t* numbers = &parserArgs.numbers[dests[i].params[0].parserParams.numbersOffset];
            numbers_[i] = new Goldilocks::Element[dests[i].params[0].parserParams.nNumbers*nrowsPack];
            for(uint64_t j = 0; j < dests[i].params[0].parserParams.nNumbers; ++j) {
                for(uint64_t k = 0; k < nrowsPack; ++k) {
                    numbers_[i][j*nrowsPack + k] = Goldilocks::fromU64(numbers[j]);
                }
            }
        }

        Goldilocks::Element publics[setupCtx.starkInfo.nPublics*nrowsPack];
        for(uint64_t i = 0; i < setupCtx.starkInfo.nPublics; ++i) {
            for(uint64_t j = 0; j < nrowsPack; ++j) {
                publics[i*nrowsPack + j] = params.publicInputs[i];
            }
        }

        Goldilocks::Element evals[setupCtx.starkInfo.evMap.size()*FIELD_EXTENSION*nrowsPack];
        for(uint64_t i = 0; i < setupCtx.starkInfo.evMap.size(); ++i) {
            for(uint64_t j = 0; j < nrowsPack; ++j) {
                evals[(i*FIELD_EXTENSION)*nrowsPack + j] = params.evals[i * FIELD_EXTENSION];
                evals[(i*FIELD_EXTENSION + 1)*nrowsPack + j] = params.evals[i * FIELD_EXTENSION + 1];
                evals[(i*FIELD_EXTENSION + 2)*nrowsPack + j] = params.evals[i * FIELD_EXTENSION + 2];
            }
        }

        Goldilocks::Element subproofValues[setupCtx.starkInfo.nSubProofValues*FIELD_EXTENSION*nrowsPack];
        for(uint64_t i = 0; i < setupCtx.starkInfo.nSubProofValues; ++i) {
            for(uint64_t j = 0; j < nrowsPack; ++j) {
                subproofValues[(i*FIELD_EXTENSION)*nrowsPack + j] = params.subproofValues[i * FIELD_EXTENSION];
                subproofValues[(i*FIELD_EXTENSION + 1)*nrowsPack + j] = params.subproofValues[i * FIELD_EXTENSION + 1];
                subproofValues[(i*FIELD_EXTENSION + 2)*nrowsPack + j] = params.subproofValues[i * FIELD_EXTENSION + 2];
            }
        }

    #pragma omp parallel for
        for (uint64_t i = 0; i < domainSize; i+= nrowsPack) {
            Goldilocks::Element bufferT_[nOpenings*nCols*nrowsPack];

            loadPolynomials(params, parserArgs, dests, bufferT_, i, domainExtended);

            Goldilocks::Element destVals[dests.size() * FIELD_EXTENSION * nrowsPack];

            for(uint64_t j = 0; j < dests.size(); ++j) {
                uint64_t i_args = 0;

                uint8_t* ops = &parserArgs.ops[dests[j].params[0].parserParams.opsOffset];
                uint16_t* args = &parserArgs.args[dests[j].params[0].parserParams.argsOffset];
            Goldilocks::Element tmp1[dests[j].params[0].parserParams.nTemp1*nrowsPack];
            Goldilocks::Element tmp3[dests[j].params[0].parserParams.nTemp3*nrowsPack*FIELD_EXTENSION];

                for (uint64_t kk = 0; kk < dests[j].params[0].parserParams.nOps; ++kk) {
                    switch (ops[kk]) {
                        case 0: {
                           // COPY commit1 to tmp1
                            Goldilocks::copy_pack(nrowsPack, &tmp1[args[i_args] * nrowsPack], &bufferT_[(nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]) * nrowsPack]);
                            i_args += 3;
                            break;
                        }
                        case 1: {
                            // OPERATION WITH DEST: tmp1 - SRC0: commit1 - SRC1: commit1
                            Goldilocks::op_pack(nrowsPack, args[i_args], &tmp1[args[i_args + 1] * nrowsPack], &bufferT_[(nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]) * nrowsPack], &bufferT_[(nColsStagesAcc[args[i_args + 4]] + args[i_args + 5]) * nrowsPack]);
                            i_args += 6;
                            break;
                        }
                        case 2: {
                            // OPERATION WITH DEST: tmp1 - SRC0: commit1 - SRC1: tmp1
                            Goldilocks::op_pack(nrowsPack, args[i_args], &tmp1[args[i_args + 1] * nrowsPack], &bufferT_[(nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]) * nrowsPack], &tmp1[args[i_args + 4] * nrowsPack]);
                            i_args += 5;
                            break;
                        }
                        case 3: {
                            // OPERATION WITH DEST: tmp1 - SRC0: commit1 - SRC1: public
                            Goldilocks::op_pack(nrowsPack, args[i_args], &tmp1[args[i_args + 1] * nrowsPack], &bufferT_[(nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]) * nrowsPack], &publics[args[i_args + 4] * nrowsPack]);
                            i_args += 5;
                            break;
                        }
                        case 4: {
                            // OPERATION WITH DEST: tmp1 - SRC0: commit1 - SRC1: number
                            Goldilocks::op_pack(nrowsPack, args[i_args], &tmp1[args[i_args + 1] * nrowsPack], &bufferT_[(nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]) * nrowsPack], &numbers_[j][args[i_args + 4]*nrowsPack]);
                            i_args += 5;
                            break;
                        }
                        case 5: {
                           // COPY tmp1 to tmp1
                            Goldilocks::copy_pack(nrowsPack, &tmp1[args[i_args] * nrowsPack], &tmp1[args[i_args + 1] * nrowsPack]);
                            i_args += 2;
                            break;
                        }
                        case 6: {
                            // OPERATION WITH DEST: tmp1 - SRC0: tmp1 - SRC1: tmp1
                            Goldilocks::op_pack(nrowsPack, args[i_args], &tmp1[args[i_args + 1] * nrowsPack], &tmp1[args[i_args + 2] * nrowsPack], &tmp1[args[i_args + 3] * nrowsPack]);
                            i_args += 4;
                            break;
                        }
                        case 7: {
                            // OPERATION WITH DEST: tmp1 - SRC0: tmp1 - SRC1: public
                            Goldilocks::op_pack(nrowsPack, args[i_args], &tmp1[args[i_args + 1] * nrowsPack], &tmp1[args[i_args + 2] * nrowsPack], &publics[args[i_args + 3] * nrowsPack]);
                            i_args += 4;
                            break;
                        }
                        case 8: {
                            // OPERATION WITH DEST: tmp1 - SRC0: tmp1 - SRC1: number
                            Goldilocks::op_pack(nrowsPack, args[i_args], &tmp1[args[i_args + 1] * nrowsPack], &tmp1[args[i_args + 2] * nrowsPack], &numbers_[j][args[i_args + 3]*nrowsPack]);
                            i_args += 4;
                            break;
                        }
                        case 9: {
                           // COPY public to tmp1
                            Goldilocks::copy_pack(nrowsPack, &tmp1[args[i_args] * nrowsPack], &publics[args[i_args + 1] * nrowsPack]);
                            i_args += 2;
                            break;
                        }
                        case 10: {
                            // OPERATION WITH DEST: tmp1 - SRC0: public - SRC1: public
                            Goldilocks::op_pack(nrowsPack, args[i_args], &tmp1[args[i_args + 1] * nrowsPack], &publics[args[i_args + 2] * nrowsPack], &publics[args[i_args + 3] * nrowsPack]);
                            i_args += 4;
                            break;
                        }
                        case 11: {
                            // OPERATION WITH DEST: tmp1 - SRC0: public - SRC1: number
                            Goldilocks::op_pack(nrowsPack, args[i_args], &tmp1[args[i_args + 1] * nrowsPack], &publics[args[i_args + 2] * nrowsPack], &numbers_[j][args[i_args + 3]*nrowsPack]);
                            i_args += 4;
                            break;
                        }
                        case 12: {
                           // COPY number to tmp1
                            Goldilocks::copy_pack(nrowsPack, &tmp1[args[i_args] * nrowsPack], &numbers_[j][args[i_args + 1]*nrowsPack]);
                            i_args += 2;
                            break;
                        }
                        case 13: {
                            // OPERATION WITH DEST: tmp1 - SRC0: number - SRC1: number
                            Goldilocks::op_pack(nrowsPack, args[i_args], &tmp1[args[i_args + 1] * nrowsPack], &numbers_[j][args[i_args + 2]*nrowsPack], &numbers_[j][args[i_args + 3]*nrowsPack]);
                            i_args += 4;
                            break;
                        }
                        case 14: {
                            // OPERATION WITH DEST: tmp3 - SRC0: commit3 - SRC1: commit1
                            Goldilocks3::op_31_pack(nrowsPack, args[i_args], &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &bufferT_[(nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]) * nrowsPack], &bufferT_[(nColsStagesAcc[args[i_args + 4]] + args[i_args + 5]) * nrowsPack]);
                            i_args += 6;
                            break;
                        }
                        case 15: {
                            // OPERATION WITH DEST: tmp3 - SRC0: commit3 - SRC1: tmp1
                            Goldilocks3::op_31_pack(nrowsPack, args[i_args], &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &bufferT_[(nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]) * nrowsPack], &tmp1[args[i_args + 4] * nrowsPack]);
                            i_args += 5;
                            break;
                        }
                        case 16: {
                            // OPERATION WITH DEST: tmp3 - SRC0: commit3 - SRC1: public
                            Goldilocks3::op_31_pack(nrowsPack, args[i_args], &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &bufferT_[(nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]) * nrowsPack], &publics[args[i_args + 4] * nrowsPack]);
                            i_args += 5;
                            break;
                        }
                        case 17: {
                            // OPERATION WITH DEST: tmp3 - SRC0: commit3 - SRC1: number
                            Goldilocks3::op_31_pack(nrowsPack, args[i_args], &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &bufferT_[(nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]) * nrowsPack], &numbers_[j][args[i_args + 4]*nrowsPack]);
                            i_args += 5;
                            break;
                        }
                        case 18: {
                            // OPERATION WITH DEST: tmp3 - SRC0: tmp3 - SRC1: commit1
                            Goldilocks3::op_31_pack(nrowsPack, args[i_args], &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &tmp3[args[i_args + 2] * nrowsPack * FIELD_EXTENSION], &bufferT_[(nColsStagesAcc[args[i_args + 3]] + args[i_args + 4]) * nrowsPack]);
                            i_args += 5;
                            break;
                        }
                        case 19: {
                            // OPERATION WITH DEST: tmp3 - SRC0: tmp3 - SRC1: tmp1
                            Goldilocks3::op_31_pack(nrowsPack, args[i_args], &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &tmp3[args[i_args + 2] * nrowsPack * FIELD_EXTENSION], &tmp1[args[i_args + 3] * nrowsPack]);
                            i_args += 4;
                            break;
                        }
                        case 20: {
                            // OPERATION WITH DEST: tmp3 - SRC0: tmp3 - SRC1: public
                            Goldilocks3::op_31_pack(nrowsPack, args[i_args], &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &tmp3[args[i_args + 2] * nrowsPack * FIELD_EXTENSION], &publics[args[i_args + 3] * nrowsPack]);
                            i_args += 4;
                            break;
                        }
                        case 21: {
                            // OPERATION WITH DEST: tmp3 - SRC0: tmp3 - SRC1: number
                            Goldilocks3::op_31_pack(nrowsPack, args[i_args], &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &tmp3[args[i_args + 2] * nrowsPack * FIELD_EXTENSION], &numbers_[j][args[i_args + 3]*nrowsPack]);
                            i_args += 4;
                            break;
                        }
                        case 22: {
                            // OPERATION WITH DEST: tmp3 - SRC0: challenge - SRC1: commit1
                            Goldilocks3::op_31_pack(nrowsPack, args[i_args], &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &challenges[args[i_args + 2]*FIELD_EXTENSION*nrowsPack], &bufferT_[(nColsStagesAcc[args[i_args + 3]] + args[i_args + 4]) * nrowsPack]);
                            i_args += 5;
                            break;
                        }
                        case 23: {
                            // OPERATION WITH DEST: tmp3 - SRC0: challenge - SRC1: tmp1
                            Goldilocks3::op_31_pack(nrowsPack, args[i_args], &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &challenges[args[i_args + 2]*FIELD_EXTENSION*nrowsPack], &tmp1[args[i_args + 3] * nrowsPack]);
                            i_args += 4;
                            break;
                        }
                        case 24: {
                            // OPERATION WITH DEST: tmp3 - SRC0: challenge - SRC1: public
                            Goldilocks3::op_31_pack(nrowsPack, args[i_args], &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &challenges[args[i_args + 2]*FIELD_EXTENSION*nrowsPack], &publics[args[i_args + 3] * nrowsPack]);
                            i_args += 4;
                            break;
                        }
                        case 25: {
                            // OPERATION WITH DEST: tmp3 - SRC0: challenge - SRC1: number
                            Goldilocks3::op_31_pack(nrowsPack, args[i_args], &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &challenges[args[i_args + 2]*FIELD_EXTENSION*nrowsPack], &numbers_[j][args[i_args + 3]*nrowsPack]);
                            i_args += 4;
                            break;
                        }
                        case 26: {
                            // OPERATION WITH DEST: tmp3 - SRC0: subproofValue - SRC1: commit1
                            Goldilocks3::op_31_pack(nrowsPack, args[i_args], &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &subproofValues[args[i_args + 2]*FIELD_EXTENSION*nrowsPack], &bufferT_[(nColsStagesAcc[args[i_args + 3]] + args[i_args + 4]) * nrowsPack]);
                            i_args += 5;
                            break;
                        }
                        case 27: {
                            // OPERATION WITH DEST: tmp3 - SRC0: subproofValue - SRC1: tmp1
                            Goldilocks3::op_31_pack(nrowsPack, args[i_args], &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &subproofValues[args[i_args + 2]*FIELD_EXTENSION*nrowsPack], &tmp1[args[i_args + 3] * nrowsPack]);
                            i_args += 4;
                            break;
                        }
                        case 28: {
                            // OPERATION WITH DEST: tmp3 - SRC0: subproofValue - SRC1: public
                            Goldilocks3::op_31_pack(nrowsPack, args[i_args], &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &subproofValues[args[i_args + 2]*FIELD_EXTENSION*nrowsPack], &publics[args[i_args + 3] * nrowsPack]);
                            i_args += 4;
                            break;
                        }
                        case 29: {
                            // OPERATION WITH DEST: tmp3 - SRC0: subproofValue - SRC1: number
                            Goldilocks3::op_31_pack(nrowsPack, args[i_args], &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &subproofValues[args[i_args + 2]*FIELD_EXTENSION*nrowsPack], &numbers_[j][args[i_args + 3]*nrowsPack]);
                            i_args += 4;
                            break;
                        }
                        case 30: {
                           // COPY commit3 to tmp3
                            Goldilocks3::copy_pack(nrowsPack, &tmp3[args[i_args] * nrowsPack * FIELD_EXTENSION], &bufferT_[(nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]) * nrowsPack]);
                            i_args += 3;
                            break;
                        }
                        case 31: {
                            // OPERATION WITH DEST: tmp3 - SRC0: commit3 - SRC1: commit3
                            Goldilocks3::op_pack(nrowsPack, args[i_args], &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &bufferT_[(nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]) * nrowsPack], &bufferT_[(nColsStagesAcc[args[i_args + 4]] + args[i_args + 5]) * nrowsPack]);
                            i_args += 6;
                            break;
                        }
                        case 32: {
                            // OPERATION WITH DEST: tmp3 - SRC0: commit3 - SRC1: tmp3
                            Goldilocks3::op_pack(nrowsPack, args[i_args], &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &bufferT_[(nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]) * nrowsPack], &tmp3[args[i_args + 4] * nrowsPack * FIELD_EXTENSION]);
                            i_args += 5;
                            break;
                        }
                        case 33: {
                            // OPERATION WITH DEST: tmp3 - SRC0: commit3 - SRC1: challenge
                            Goldilocks3::op_pack(nrowsPack, args[i_args], &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &bufferT_[(nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]) * nrowsPack], &challenges[args[i_args + 4]*FIELD_EXTENSION*nrowsPack]);
                            i_args += 5;
                            break;
                        }
                        case 34: {
                            // OPERATION WITH DEST: tmp3 - SRC0: commit3 - SRC1: subproofValue
                            Goldilocks3::op_pack(nrowsPack, args[i_args], &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &bufferT_[(nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]) * nrowsPack], &subproofValues[args[i_args + 4]*FIELD_EXTENSION*nrowsPack]);
                            i_args += 5;
                            break;
                        }
                        case 35: {
                           // COPY tmp3 to tmp3
                            Goldilocks3::copy_pack(nrowsPack, &tmp3[args[i_args] * nrowsPack * FIELD_EXTENSION], &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION]);
                            i_args += 2;
                            break;
                        }
                        case 36: {
                            // OPERATION WITH DEST: tmp3 - SRC0: tmp3 - SRC1: tmp3
                            Goldilocks3::op_pack(nrowsPack, args[i_args], &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &tmp3[args[i_args + 2] * nrowsPack * FIELD_EXTENSION], &tmp3[args[i_args + 3] * nrowsPack * FIELD_EXTENSION]);
                            i_args += 4;
                            break;
                        }
                        case 37: {
                            // OPERATION WITH DEST: tmp3 - SRC0: tmp3 - SRC1: challenge
                            Goldilocks3::op_pack(nrowsPack, args[i_args], &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &tmp3[args[i_args + 2] * nrowsPack * FIELD_EXTENSION], &challenges[args[i_args + 3]*FIELD_EXTENSION*nrowsPack]);
                            i_args += 4;
                            break;
                        }
                        case 38: {
                            // OPERATION WITH DEST: tmp3 - SRC0: tmp3 - SRC1: subproofValue
                            Goldilocks3::op_pack(nrowsPack, args[i_args], &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &tmp3[args[i_args + 2] * nrowsPack * FIELD_EXTENSION], &subproofValues[args[i_args + 3]*FIELD_EXTENSION*nrowsPack]);
                            i_args += 4;
                            break;
                        }
                        case 39: {
                            // OPERATION WITH DEST: tmp3 - SRC0: challenge - SRC1: challenge
                            Goldilocks3::op_pack(nrowsPack, args[i_args], &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &challenges[args[i_args + 2]*FIELD_EXTENSION*nrowsPack], &challenges[args[i_args + 3]*FIELD_EXTENSION*nrowsPack]);
                            i_args += 4;
                            break;
                        }
                        case 40: {
                            // OPERATION WITH DEST: tmp3 - SRC0: challenge - SRC1: subproofValue
                            Goldilocks3::op_pack(nrowsPack, args[i_args], &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &challenges[args[i_args + 2]*FIELD_EXTENSION*nrowsPack], &subproofValues[args[i_args + 3]*FIELD_EXTENSION*nrowsPack]);
                            i_args += 4;
                            break;
                        }
                        case 41: {
                            // OPERATION WITH DEST: tmp3 - SRC0: subproofValue - SRC1: subproofValue
                            Goldilocks3::op_pack(nrowsPack, args[i_args], &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &subproofValues[args[i_args + 2]*FIELD_EXTENSION*nrowsPack], &subproofValues[args[i_args + 3]*FIELD_EXTENSION*nrowsPack]);
                            i_args += 4;
                            break;
                        }
                        case 42: {
                           // COPY eval to tmp3
                            Goldilocks3::copy_pack(nrowsPack, &tmp3[args[i_args] * nrowsPack * FIELD_EXTENSION], &evals[args[i_args + 1]*FIELD_EXTENSION*nrowsPack]);
                            i_args += 2;
                            break;
                        }
                        case 43: {
                            // OPERATION WITH DEST: tmp3 - SRC0: challenge - SRC1: eval
                            Goldilocks3::op_pack(nrowsPack, args[i_args], &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &challenges[args[i_args + 2]*FIELD_EXTENSION*nrowsPack], &evals[args[i_args + 3]*FIELD_EXTENSION*nrowsPack]);
                            i_args += 4;
                            break;
                        }
                        case 44: {
                            // OPERATION WITH DEST: tmp3 - SRC0: tmp3 - SRC1: eval
                            Goldilocks3::op_pack(nrowsPack, args[i_args], &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &tmp3[args[i_args + 2] * nrowsPack * FIELD_EXTENSION], &evals[args[i_args + 3]*FIELD_EXTENSION*nrowsPack]);
                            i_args += 4;
                            break;
                        }
                        case 45: {
                            // OPERATION WITH DEST: tmp3 - SRC0: eval - SRC1: commit1
                            Goldilocks3::op_31_pack(nrowsPack, args[i_args], &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &evals[args[i_args + 2]*FIELD_EXTENSION*nrowsPack], &bufferT_[(nColsStagesAcc[args[i_args + 3]] + args[i_args + 4]) * nrowsPack]);
                            i_args += 5;
                            break;
                        }
                        case 46: {
                            // OPERATION WITH DEST: tmp3 - SRC0: commit3 - SRC1: eval
                            Goldilocks3::op_pack(nrowsPack, args[i_args], &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &bufferT_[(nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]) * nrowsPack], &evals[args[i_args + 4]*FIELD_EXTENSION*nrowsPack]);
                            i_args += 5;
                            break;
                        }
                        default: {
                            std::cout << " Wrong operation!" << std::endl;
                            exit(1);
                        }
                    }
                }

                if (i_args != dests[j].params[0].parserParams.nArgs) std::cout << " " << i_args << " - " << dests[j].params[0].parserParams.nArgs << std::endl;
                assert(i_args == dests[j].params[0].parserParams.nArgs);

                copyPolynomial(&destVals[j*FIELD_EXTENSION*nrowsPack], dests[j].params[0].inverse, dests[j].params[0].parserParams.destId, dests[j].params[0].parserParams.destDim, tmp1, tmp3);
            }
            storePolynomial(dests, destVals, i);

        }
        for (uint64_t i = 0; i < dests.size(); ++i) {
            delete numbers_[i];
        }

        delete[] numbers_;
    }
};

#endif