
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_lda_c_chachiyo_BrOH_cation_2_zk():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_c_chachiyo", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.644049156573815e-01, -1.644050013831216e-01, -1.644054089559213e-01, -1.644041329566611e-01, -1.644047977789955e-01, -1.644047977789955e-01, -1.096911927482342e-01, -1.096912556231132e-01, -1.096934227865733e-01, -1.096951867312902e-01, -1.096919562755926e-01, -1.096919562755926e-01, -6.525800332746322e-02, -6.523092092786574e-02, -6.458163825185124e-02, -6.478275460089096e-02, -6.449405314054367e-02, -6.449405314054367e-02, -3.679154044580768e-02, -3.704504938176947e-02, -6.894847705668358e-02, -3.212315946378072e-02, -2.677079448138688e-02, -2.677079448138689e-02, -3.495694045108312e-03, -3.658718535522795e-03, -1.454028037037199e-02, -2.108489125768041e-03, -2.223866876571491e-03, -2.223866876571491e-03, -1.215769719402438e-01, -1.215824295959175e-01, -1.215772435578623e-01, -1.215820618606107e-01, -1.215797273908245e-01, -1.215797273908245e-01, -9.327625751658923e-02, -9.347788843812344e-02, -9.312796130948477e-02, -9.330705614800684e-02, -9.345729793266569e-02, -9.345729793266569e-02, -6.206800151413210e-02, -6.403367156971558e-02, -6.018418451061213e-02, -6.109807566046104e-02, -6.230032569464852e-02, -6.230032569464849e-02, -2.644777994191731e-02, -3.686335942080335e-02, -2.526410778767834e-02, -9.273824570338325e-02, -2.878261194773572e-02, -2.878261194773572e-02, -1.647627007169225e-03, -2.064011126471297e-03, -1.596672028734686e-03, -2.001676607024458e-02, -1.798204863061205e-03, -1.798204863061205e-03, -6.218166121767515e-02, -6.204614885359534e-02, -6.209390040401833e-02, -6.213318758721104e-02, -6.211352943152492e-02, -6.211352943152492e-02, -6.153595418667166e-02, -5.800959314500775e-02, -5.906273671882454e-02, -6.007085870691112e-02, -5.956052361809609e-02, -5.956052361809609e-02, -6.516779035404831e-02, -4.114705712107020e-02, -4.468322607856446e-02, -5.037374142875949e-02, -4.747354218602754e-02, -4.747354218602754e-02, -5.614299129359848e-02, -1.413566025633532e-02, -1.741253073302816e-02, -4.959496508895527e-02, -2.310296032214484e-02, -2.310296032214484e-02, -4.760226466713622e-03, -5.763409976102291e-04, -1.185442004703973e-03, -2.230896732165254e-02, -1.689209516080052e-03, -1.689209516080049e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_lda_c_chachiyo_BrOH_cation_2_vrho():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_c_chachiyo", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-1.746508924905330e-01, -1.746514204480951e-01, -1.746507973792044e-01, -1.746516876537934e-01, -1.746521437475712e-01, -1.746511594795010e-01, -1.746488093382315e-01, -1.746519323793726e-01, -1.746484254390609e-01, -1.746536509571819e-01, -1.746484254390609e-01, -1.746536509571819e-01, -1.194003469293464e-01, -1.194040263669870e-01, -1.194001500170474e-01, -1.194043515516978e-01, -1.194072370080604e-01, -1.194016851456783e-01, -1.194081524467905e-01, -1.194043676555558e-01, -1.194534347128825e-01, -1.193525504914763e-01, -1.194534347128825e-01, -1.193525504914763e-01, -7.361187999005035e-02, -7.299533506394304e-02, -7.365426361303863e-02, -7.289735110511103e-02, -7.205755506029581e-02, -7.312298891330268e-02, -7.296600952028745e-02, -7.263137674748286e-02, -6.791885980640988e-02, -7.801790622222475e-02, -6.791885980640988e-02, -7.801790622222475e-02, -4.438931037989959e-02, -4.119451631358676e-02, -4.492615375118501e-02, -4.126345673477094e-02, -8.006944067319675e-02, -7.460076648515900e-02, -3.810353826259670e-02, -3.711640541704217e-02, -2.537303155685420e-02, -6.814988014947579e-02, -2.537303155685422e-02, -6.814988014947577e-02, -4.761085584127224e-03, -4.376563850851912e-03, -5.010853053205764e-03, -4.553373877263849e-03, -1.877267963616430e-02, -1.714610659117271e-02, -2.739838854726868e-03, -2.802394793213640e-03, -2.535247631778200e-03, -4.922474993259182e-03, -2.535247631778200e-03, -4.922474993259182e-03, -1.314518280389225e-01, -1.315237346556926e-01, -1.314564127445986e-01, -1.315302198058276e-01, -1.314514347459004e-01, -1.315246797460554e-01, -1.314568697284480e-01, -1.315290158603058e-01, -1.314539628258630e-01, -1.315271890484233e-01, -1.314539628258630e-01, -1.315271890484233e-01, -1.025671215993507e-01, -1.025786220818025e-01, -1.027497666043004e-01, -1.028118509934974e-01, -1.026167313629687e-01, -1.022240980563667e-01, -1.028100871676857e-01, -1.024002462003672e-01, -1.022622806337049e-01, -1.032631966218568e-01, -1.022622806337049e-01, -1.032631966218568e-01, -6.972119099073400e-02, -7.011275195058596e-02, -7.203845215680793e-02, -7.196913541951321e-02, -7.062780742808512e-02, -6.546455115900557e-02, -7.137176051182216e-02, -6.662165627086120e-02, -6.664658809916893e-02, -7.423984346818263e-02, -6.664658809916892e-02, -7.423984346818259e-02, -3.162441707747707e-02, -3.107348899062637e-02, -4.299296282837776e-02, -4.257566503771785e-02, -3.158506605477903e-02, -2.869349217252172e-02, -1.019671082616394e-01, -1.020687581705889e-01, -3.591905613126201e-02, -3.223638672136684e-02, -3.591905613126201e-02, -3.223638672136684e-02, -2.230241704355095e-03, -2.119184366015677e-03, -2.740663202203366e-03, -2.685965851133672e-03, -2.192076411908183e-03, -2.031761668354857e-03, -2.428747518191940e-02, -2.402661234907369e-02, -3.109160646124947e-03, -2.077048892527144e-03, -3.109160646124947e-03, -2.077048892527143e-03, -7.032413280059685e-02, -6.975314201067720e-02, -7.018230294164986e-02, -6.960691739151649e-02, -7.023320374948996e-02, -6.965754563918006e-02, -7.027250554794708e-02, -6.970171514327243e-02, -7.025281050227614e-02, -6.967964329437260e-02, -7.025281050227614e-02, -6.967964329437260e-02, -6.959308698263258e-02, -6.911011241228388e-02, -6.589140864687951e-02, -6.529764920066977e-02, -6.702838980170317e-02, -6.640829051477574e-02, -6.804241867891254e-02, -6.754185055952072e-02, -6.751043268407352e-02, -6.698649972023490e-02, -6.751043268407352e-02, -6.698649972023490e-02, -7.333324194503953e-02, -7.307986426865752e-02, -4.771697846843135e-02, -4.717092355291935e-02, -5.179630057712297e-02, -5.076335503692553e-02, -5.773757722410814e-02, -5.708736014730392e-02, -5.426652645641489e-02, -5.430553595705424e-02, -5.426652645641490e-02, -5.430553595705424e-02, -6.405757689118440e-02, -6.314763442169569e-02, -1.752632534206023e-02, -1.732941024188555e-02, -2.182733650550930e-02, -2.062719762911889e-02, -5.745838626525107e-02, -5.572287123547608e-02, -2.908833317977997e-02, -2.636791884943495e-02, -2.908833317977998e-02, -2.636791884943495e-02, -6.314712266309286e-03, -6.001845937727218e-03, -7.664146887129283e-04, -7.641608454896113e-04, -1.636608548579720e-03, -1.509706022963514e-03, -2.713286264957761e-02, -2.635373005347561e-02, -2.828337810963792e-03, -1.965812245396334e-03, -2.828337810963788e-03, -1.965812245396332e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05