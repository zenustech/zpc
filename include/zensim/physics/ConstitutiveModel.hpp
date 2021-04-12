#pragma once

namespace zs {

  enum constitutive_model_e : char {
    JBased = 0,
    NeoHookean,
    FixedCorotated,
    DruckerPrager,
    NACC,
    NumConstitutiveModels
  };

}