package unocard;

import game.SpecialEffectsManager;

public class SkipCard extends ActionCard {
	
	public SkipCard(Color color) {
		super(color, ActionCard.Content.SKIP);
	}
	
	
	/**
	 * Next player in sequence misses a turn
	 * @param effManager: the object that controls the game effects.
	 */
	@Override
	public void doSpecialEffect(SpecialEffectsManager effManager){
		effManager.doSkip();
	}
	
}
